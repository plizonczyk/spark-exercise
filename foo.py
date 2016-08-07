import pprint as pp
import random
import math


def main():
    import findspark
    findspark.init()

    import pyspark
    sc = pyspark.SparkContext()

    # Mute spark logging
    apache_logger = sc._jvm.org.apache.log4j
    apache_logger.LogManager.getLogger("org").setLevel(apache_logger.Level.ERROR)
    apache_logger.LogManager.getLogger("akka").setLevel(apache_logger.Level.ERROR)

    # Create RDD
    floatRdd = sc.textFile("data/yeast_no_header_data.txt")\
                 .map(lambda s: s.split('\t'))\
                 .map(lambda s: list(map(float, s[1:])))
    floatRdd.cache()

    # reduce dimensions
    rowSize = len(floatRdd.first())

    def getDimset(max_index):
        workingDims = random.randint(2, max_index)
        indices = set()
        for _ in range(workingDims):
            indices.add(random.randint(0, max_index))
        return indices

    def mutateDimensionSet(rowSize, factor):
        def mutationImpl(dimset):
            currDimAmount = len(dimset)
            # print('currDimAmount:', currDimAmount)
            mutationMaxAmount = math.floor(currDimAmount * factor)
            # print('mutationMaxAmount:', mutationMaxAmount)
            mutation_diff = getDimset(mutationMaxAmount)
            # print('diff_set')
            # pp.pprint(mutation_diff, compact=True)
            mutation_sum = getDimset(mutationMaxAmount)
            # print('sumset')
            # pp.pprint(mutation_sum, compact=True)
            return (dimset - mutation_diff) | mutation_sum
        return mutationImpl

    mutations_amount = 5
    # relative amount of mutations
    factor = 0.2
    # Standard abb
    standard_dev = 0.2
    # amount of centroids
    centroids_amount = 10
    # amount of centroid recalculations
    n = 2

    # get seed dimension set
    seedDimset = getDimset(rowSize)
    workingDimsAmount = len(seedDimset)
    print('Working dims amount:', seedDimset)
    pp.pprint(seedDimset, compact=True)
    mutator = mutateDimensionSet(rowSize, factor)
    curr_dimset = seedDimset
    dimSets = [seedDimset]
    for index in range(mutations_amount):
        new_dimset = mutator(curr_dimset)
        print('Changed indices, mutaton: ', index)
        print('Added:')
        pp.pprint(new_dimset - curr_dimset, compact=True)
        print('Deleted:')
        pp.pprint((curr_dimset - new_dimset), compact=True)
        curr_dimset = new_dimset
        print('new_size:', len(curr_dimset))
        dimSets.append(curr_dimset)

    def workOnDimsetClosure(dimSets, floatRdd, centroids_amount, standard_dev, n):
        def closureImpl(index):
            return (index, workOnDimset(dimSets[index], floatRdd, centroids_amount, standard_dev, n))
        return closureImpl

    biclusteringWorker = workOnDimsetClosure(dimSets, floatRdd, centroids_amount, standard_dev, n)

    from multiprocessing.pool import ThreadPool
    tpool = ThreadPool(processes=mutations_amount)
    results = tpool.map(biclusteringWorker, range(len(dimSets)))

    # results = []
    # for index in range(len(dimSets)):
    #     results.append((index, workOnDimset(dimSets[index], floatRdd, centroids_amount, standard_dev, n)))
    results.sort(key=lambda x: x[1])
    print('(Mutation index, (quality, rows amount, cols amount, rdd, centroid))')
    for result in results:
        print(result)


def workOnDimset(workingDims, floatRdd, centroidsAmount, standard_dev, n):
    dimReducedRdd = floatRdd.map(lambda a: [a[dim] for dim in range(len(a)) if dim in workingDims])
    dimReducedRdd.cache()
    # print('Reduced rowsize:', len(dimReducedRdd.first()))

    # take random seeds
    centroids = dimReducedRdd.takeSample(False, centroidsAmount)
    # print('sample:')
    # pp.pprint(centroids, compact=True)
    # print("Sample:", len(centroids))

    # bound helpers
    lowerBound = 1 - standard_dev
    upperBound = 1 + standard_dev

    def getFilterFunction(dim_boundaries):
        def filter_by_bounds(record):
            suspect = max(zip(record, dim_boundaries), key=lambda x: x[0])
            dim, bounds = suspect
            lower, upper = bounds
            return lower < dim < upper
        return filter_by_bounds

    # amount of centroid recalculations
    filteredRdds = []
    for _ in range(n):
        # calculate boundaries for each centroid, filter records
        currFilteredRdds = []
        for i in range(centroidsAmount):
            dimBounds = [(col * lowerBound, col * upperBound) for col in centroids[i]]
            filterByBounds = getFilterFunction(dimBounds)
            currFilteredRdds.append(dimReducedRdd.filter(filterByBounds))

        filteredRdds = currFilteredRdds

        # calculate new centroids
        for _ in range(centroidsAmount):
            rdd = filteredRdds[i]
            length = rdd.count()
            try:
                mean = rdd.reduce(lambda a, b: [(x + y) for x, y in zip(a, b)])
                centroids[i] = [x/length for x in mean]
            except ValueError:
                print('Rdd id:', i, 'is empty')
        # print('Centroid length', len(centroids[0]))

    qualities = []
    for rdd, centroid in zip(filteredRdds, centroids):
        rowsAmount = rdd.count()
        colsAmount = len(rdd.first()) if rowsAmount != 0 else 0
        quality = colsAmount * rowsAmount
        qualities.append((quality, rowsAmount, colsAmount, rdd, centroid))
    qualities.sort(key=lambda x: x[0], reverse=True)
    # pp.pprint(qualities, compact=True)
    return qualities[0]
    # print(len(centroids))
    # pp.pprint(centroids)

if __name__ == "__main__":
    main()

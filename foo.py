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
    standard_dev = 0.3
    # amount of centroids
    centroids_amount = 30
    # amount of centroid recalculations
    n = 2

    # get seed dimension set
    seedDimset = getDimset(rowSize)
    workingDimsAmount = len(seedDimset)
    print('Base dimSet:', seedDimset)
    pp.pprint(seedDimset, compact=True)
    mutator = mutateDimensionSet(rowSize, factor)
    dimSets = [seedDimset]
    for index in range(mutations_amount):
        curr_dimset = dimSets[-1]
        new_dimset = mutator(curr_dimset)
        print('Mutation {} changed: '.format(index))
        print('Added: ', end="")
        pp.pprint(new_dimset - curr_dimset, compact=True)
        print('Deleted: ', end="")
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

    results.sort(key=lambda x: x[1][0])
    print('(Mutation index, (quality, rows amount, cols amount, rdd, centroid))')
    for result in results:
        print(result)


def workOnDimset(workingDims, floatRdd, centroids_amount, standard_dev, n):
    dimReducedRdd = floatRdd.map(lambda a: [a[dim] for dim in range(len(a)) if dim in workingDims])
    dimReducedRdd.cache()
    # print('Reduced rowsize:', len(dimReducedRdd.first()))

    # take random centroids (seeds)
    centroids = dimReducedRdd.takeSample(False, centroids_amount)
    # print('sample:')
    # pp.pprint(centroids, compact=True)
    # print("Sample:", len(centroids))

    # bound helpers
    lowerBound = 1 - standard_dev
    upperBound = 1 + standard_dev

    def getFilterFunction(centroid, standard_dev):
        def filterImpl(record):
            params = [(centr_dim, abs(rec_dim - centr_dim)) for rec_dim, centr_dim in zip(record, centroid)]
            suspect = max(params, key=lambda x: x[1])
            dim, diff = suspect
            return diff <= abs(dim * standard_dev)

        return filterImpl

    # amount of centroid recalculations
    filteredRdds = []
    for _ in range(n):
        # calculate boundaries for each centroid, filter records
        currFilteredRdds = []
        for i in range(centroids_amount):
            filterByBounds = getFilterFunction(centroids[i], standard_dev)
            currFilteredRdds.append(dimReducedRdd.filter(filterByBounds))

        filteredRdds = currFilteredRdds

        # calculate new centroids
        for _ in range(centroids_amount):
            rdd = filteredRdds[i]
            length = rdd.count()
            try:
                mean = rdd.reduce(lambda a, b: [(x + y) for x, y in zip(a, b)])
                centroids[i] = [x/length for x in mean]
            except ValueError:
                print('Rdd id:', i, 'is empty')

    qualities = []
    for rdd, centroid in zip(filteredRdds, centroids):
        rowsAmount = rdd.count()
        colsAmount = len(rdd.first()) if rowsAmount != 0 else 0
        quality = colsAmount * rowsAmount
        qualities.append((quality, rowsAmount, colsAmount, rdd, centroid))
    qualities.sort(key=lambda x: x[0], reverse=True)
    return qualities[0]

if __name__ == "__main__":
    main()

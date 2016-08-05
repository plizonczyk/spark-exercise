import pprint as pp
import functools as ft
import random
from utils import euclidean_distance


def main():
    """basic concept, just one cluster"""
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

    def getCurrentDims(rowSize):
        print('Rowsize:', rowSize)
        workingDims = random.randint(2, rowSize)
        indices = set()
        for _ in range(workingDims):
            indices.add(random.randint(0, rowSize))
        return indices

    workingDims = getCurrentDims(rowSize)
    print('Working dims:')
    pp.pprint(workingDims, compact=True)
    workingDimsAmount = len(workingDims)
    print('Working dims amount:', workingDimsAmount)

    dimReducedRdd = floatRdd.map(lambda a: [a[i] for i in range(len(a)) if i in workingDims])
    dimReducedRdd.cache()
    print('Reduced rowsize:', len(dimReducedRdd.first()))

    # amount of runs
    quantity = 10
    # take random seeds
    centroids = dimReducedRdd.takeSample(False, quantity)
    # print('sample:')
    # pp.pprint(centroids, compact=True)
    # print("Sample:", len(centroids))

    # bound helpers
    tolerance = 0.2
    lowerBound = 1 - tolerance
    upperBound = 1 + tolerance

    def getFilterFunction(dim_boundaries):
        def filter_by_bounds(record):
            suspect = max(zip(record, dim_boundaries), key=lambda x: x[0])
            dim, bounds = suspect
            lower, upper = bounds
            return lower < dim < upper
        return filter_by_bounds

    # amount of centroid recalculations
    n = 2
    filteredRdds = []
    for _ in range(n):
        # calculate boundaries for each centroid, filter records
        currFilteredRdds = []
        for i in range(quantity):
            dimBounds = [(col * lowerBound, col * upperBound) for col in centroids[i]]
            filterByBounds = getFilterFunction(dimBounds)
            currFilteredRdds.append(dimReducedRdd.filter(filterByBounds))

        filteredRdds = currFilteredRdds

        # calculate new centroids
        for _ in range(quantity):
            rdd = filteredRdds[i]
            length = rdd.count()
            try:
                mean = rdd.reduce(lambda a, b: [(x + y) for x, y in zip(a, b)])
                centroids[i] = [x/length for x in mean]
            except ValueError:
                print('Rdd id:', i, 'is empty')
        print('Centroid length', len(centroids[0]))

    qualities = []
    for rdd in filteredRdds:
        rowsAmount = rdd.count()
        colsAmount = len(rdd.first()) if rowsAmount != 0 else 0
        quality = colsAmount * rowsAmount
        qualities.append((rdd, quality, rowsAmount, colsAmount))
    qualities.sort(key=lambda x: x[1], reverse=True)
    pp.pprint(qualities)
    cache = []
    cache.append[qualities[0]]
    # print(len(centroids))
    # pp.pprint(centroids)

if __name__ == "__main__":
    main()

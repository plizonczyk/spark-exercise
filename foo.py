import pprint as pp
import functools as ft
import random
from utils import euclidean_norm


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
    print('Reduced rowsize:', len(dimReducedRdd.first()))

    # map to scalars
    meansRDD = dimReducedRdd.map(euclidean_norm)
    print('Records:', meansRDD.count())

    # take random seeds mean, should be reworked to use them concurrently instead
    quantity = 10
    sample = meansRDD.takeSample(False, quantity)
    print('sample:')
    pp.pprint(sample)
    sampleMean = ft.reduce(lambda a, b: a + b, sample) / quantity
    print('sampleMean', sampleMean)

    tolerance = 0.2
    upperBound = (1 + tolerance) * sampleMean
    lowerBound = (1 - tolerance) * sampleMean
    print('bounds [', lowerBound, ',', upperBound, ']')

    # Filter records
    sampleRange = meansRDD.filter(lambda a: lowerBound <= a and a <= upperBound)
    sampleAmount = sampleRange.count()
    qualityInd = sampleAmount * workingDimsAmount
    print('Filtered samples:', sampleAmount)
    print('Quality ind:', qualityInd)
    meanAfterIteration = sampleRange.reduce(lambda a, b: a + b) / sampleAmount
    print("new mean:", meanAfterIteration)


if __name__ == "__main__":
    main()
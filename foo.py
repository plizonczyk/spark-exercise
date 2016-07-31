import pprint as pp
import functools as ft
from utils import euclidean_norm


def main():
    """basic concept, without reducing dimensions, parallel run"""
    import findspark
    findspark.init()

    import pyspark
    sc = pyspark.SparkContext()

    # Mute spark logging
    apache_logger = sc._jvm.org.apache.log4j
    apache_logger.LogManager.getLogger("org").setLevel(apache_logger.Level.ERROR)
    apache_logger.LogManager.getLogger("akka").setLevel(apache_logger.Level.ERROR)

    # Create RDD
    textFile = sc.textFile("data/yeast_no_header_data.txt")\
                 .map(lambda s: s.split('\t'))\
                 .map(lambda s: list(map(float, s[1:])))
    textFile.cache()

    # def getMean(data_tuple):
    #     name, characteristics = data_tuple
    #     return name, euclidean_norm(characteristics)

    # map to scalars, reduce dimensions as well?
    meansRDD = textFile.map(euclidean_norm)
    print('Records:', meansRDD.count())

    # take random mean
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

    sampleRange = meansRDD.filter(lambda a: lowerBound <= a and a <= upperBound)
    sampleAmount = sampleRange.count()
    print('Filtered samples:', sampleAmount)
    meanAfterIteration = sampleRange.reduce(lambda a, b: a + b) / sampleAmount
    print("new mean:", meanAfterIteration)


if __name__ == "__main__":
    main()
import utils
from utils import VectorWithNorm
from pprint import pprint


class KMeans:
    def __init__(self):
        self.k = 0
        self.runs = 0
        self.max_iterations = None
        # self.epsilon = epsilon

    def train(self, rdd, runs):
        self.runs = runs
        self._run(rdd)

    def _run(self, rdd):
        norms = rdd.map(lambda w: utils.euclidean_norm(w))
        norms.persist()
        # pprint(norms.collect())

        zipped_data = rdd.zip(norms).map(lambda w: VectorWithNorm(w[0], w[1]))
        # pprint(zipped_data.collect())
        # TODO: main loop
        self._run_algorithm(zipped_data)
        # model = self._run_algorithm(zipped_data)
        norms.unpersist()

    def _run_algorithm(self, rdd):
        sc = rdd.SparkContext

        if not self.runs or not self.k:
            raise RuntimeError("0 value is not valid for params")

        # TODO: explore other init methods
        self._random_init(rdd)

    def _random_init(self, rdd):
        # Spark uses xorbitshift, im lazy so default randomness source
        sample = rdd.takeSample(True, self.runs * self.k)



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
    textFile = sc.textFile("train.txt")
    splitRDD = textFile.map(lambda w: w.split('\t'))
    floatRDD = splitRDD.map(lambda w: [float(x) for x in w[0:-1]])

    means = KMeans()
    means.train(floatRDD)

    # KMeans(floatRDD, 3, max_iterations=10, runs=10)

if __name__ == "__main__":
    main()

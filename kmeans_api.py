import utils
from utils import VectorWithNorm
from pprint import pprint


class KMeans:
    def __init__(self):
        self.k = 0
        self.runs = 0
        self.max_iterations = 0
        # self.epsilon = epsilon

    def train(self, rdd, k, runs, max_iterations):
        self.runs = runs
        self.max_iterations = max_iterations
        self.k = k
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
        sc = rdd.context

        if not self.runs or not self.k or not self.max_iterations:
            raise RuntimeError("0 value is not valid for params")

        # TODO: explore other init methods
        centers = self._random_init(rdd)

        active = [True for _ in range(self.runs)]
        costs = [0 for _ in range(self.runs)]

        active_runs = [0 for _ in range(self.runs)]
        iteration = 0
        pprint(active_runs)
        # while iteration < self.max_iterations and active_runs:

        active_centers = list(map(lambda w: centers[w], active_runs))
        cost_accumulators = list(map(lambda _: sc.accumulator(0.0), active_runs))
        pprint(active_centers)
        pprint(cost_accumulators)
        broadcast_active_centers = sc.broadcast(active_centers)

        def fetch_sum_and_count_for_each_center(points):
            this_active_centers = broadcast_active_centers.value
            runs = len(this_active_centers)
            k = len(this_active_centers[0])
            dims = len(this_active_centers[0][0].vec)

            sums = [[[0 for _ in range(dims)] for _ in range(k)] for _ in range(runs)]
            counts = [[0 for _ in range(k)] for _ in range(runs)]

            def populate(point):
                for i in range(runs):
                    active_centers = this_active_centers(i)
                    (best_center, cost) = find_closest(active_centers, point)
                    cost_accumulators[i] += cost
                    # the hell are 2 lines below for
                    sum = sums[i][best_center]
                    point.vec = [x + y for x, y in zip(point.vec, sum)]
                    counts[i][best_center] += 1

            points.foreach(populate)

        total_contributions = rdd.mapPartitions(fetch_sum_and_count_for_each_center)

    def _random_init(self, rdd):
        # Spark uses xorbitshift, im lazy so default randomness source
        sample = rdd.takeSample(True, self.runs * self.k)

        inits = [sample[r * self.k:(r+1) * self.k] for r in range(self.runs)]
        return inits


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
    means.train(floatRDD, k=3, runs=10, max_iterations=10)

    # KMeans(floatRDD, 3, max_iterations=10, runs=10)

if __name__ == "__main__":
    main()

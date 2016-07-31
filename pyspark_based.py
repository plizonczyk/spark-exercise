import findspark

findspark.init()

import pyspark
from pyspark.mllib.clustering import KMeans
sc = pyspark.SparkContext()

# Mute spark logging
apache_logger = sc._jvm.org.apache.log4j
apache_logger.LogManager.getLogger("org").setLevel(apache_logger.Level.ERROR)
apache_logger.LogManager.getLogger("akka").setLevel(apache_logger.Level.ERROR)

# Create RDD
textFile = sc.textFile("data/train.txt")
splitRDD = textFile.map(lambda w: w.split('\t'))
floatRDD = splitRDD.map(lambda w: [float(x) for x in w[0:-1]])

# Build the model (cluster the data)
clusters = KMeans.train(floatRDD, 3, maxIterations=10, runs=10, initializationMode="random")

# Load test data
testData = []
with open('test.txt') as f:
    for line in f:
        split_line = line.split('\t')
        testData.append(list(map(float, split_line[0:-1])) + [split_line[-1]])

# Run predictions
for test in testData:
    print("Class", test[-1].strip('\n'), "prediction:", clusters.predict(test[0:-1]))

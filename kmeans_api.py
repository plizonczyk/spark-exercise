import findspark
findspark.init()

from pprint import pprint
from pyspark import SparkContext

sc = SparkContext()

from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt

# Load and parse the data
data = sc.textFile("data.txt")
splitFile = data.map(lambda w: w.split('\t'))
floatFile = splitFile.map(lambda w: [float(x) for x in w[0:-1]] + [w[-1]])
# parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))
parsedData = floatFile.map(lambda w: array(w[0:-1]))
pprint(parsedData.collect())

# Build the model (cluster the data)
clusters = KMeans.train(parsedData, 2, maxIterations=10,
        runs=10, initializationMode="random")

# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

#
# # Save and load model
# clusters.save(sc, "myModelPath")
# sameModel = KMeansModel.load(sc, "myModelPath")
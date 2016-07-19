import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()

from pprint import pprint
from matplotlib import pyplot

import math

textFile = sc.textFile("data.txt")
splitFile = textFile.map(lambda w: w.split('\t'))
floatFile = splitFile.map(lambda w: [float(x) for x in w[0:-1]] + [w[-1]])
# pprint(floatFile.collect())
versicolor = floatFile.filter(lambda row: row[-1] == "Iris-versicolor")

def cartesian_distance(a, b):
    distances = [(z - y)**2 for z, y in zip(a, b)]
    return math.sqrt(sum(distances))

# print(cartesian_distance([3, 4], [0,0]))


pruned = versicolor.map(lambda row: row[0:-1])
sample = pruned.take(1)
distances = pruned.map(lambda row: cartesian_distance(row, sample))
# pprint(pruned.collect())
pprint(distances.collect())

# print(textFile.count())
# pprint(filtered.collect())

# linesWithSpark = textFile.filter(lambda line: "third" in line)
# linesWithSpark.count()

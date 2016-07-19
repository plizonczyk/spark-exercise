import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()

from pprint import pprint

textFile = sc.textFile("data.txt")
splitFile = textFile.map(lambda w: w.split('\t'))
filtered = splitFile.filter(lambda row: row[-1] == "Iris-versicolor")
pprint(filtered.collect())
# print(textFile.count())

# linesWithSpark = textFile.filter(lambda line: "third" in line)
# linesWithSpark.count()

import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()

textFile = sc.textFile("data.txt")
print(textFile.count())

linesWithSpark = textFile.filter(lambda line: "third" in line)
linesWithSpark.count()

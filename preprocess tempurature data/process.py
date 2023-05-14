import numpy as np
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("processer").getOrCreate()

cols_to_keep = ['site', 'site_id', 'subsite', 'subsite_id', 'depth', 'parameter', 'lat', 'lon', 'gbrmpa_reef_id', 'time', 'cal_val', 'qc_val', 'qc_flag']

df = spark.read.csv("short.csv", header=True)

df = df.select(cols_to_keep)

df.show(5)

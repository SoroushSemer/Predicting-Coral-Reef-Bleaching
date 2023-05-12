#################################################
#                                               #
#   Team Tidal                                  #
#   Kelvin Chen, Daniel Laszczych,              #
#   Stanley Wong, Soroush Semerkant             #
#                                               #
#   This is the script for Hypothesis Testing   #
#   using PySpark run on GCP Cluster            # 
#                                               #
#################################################

import sys
from pprint import pprint

import random
import numpy as np
import math


from scipy.stats import t as tdist # this is solely to use for the cdf of the t distribution, pearson corr is calculated manually 

from pyspark.sql import SparkSession
from pyspark.context import SparkContext

random.seed(0)



sc = SparkContext()
sc.setLogLevel("ERROR")

def rdd_print(rdd, val:int):
    vals = rdd.take(val)
    for i in vals:
        print(i)


filename = sys.argv[1] #the data file to read into a stream

#read the file into the rdd
rddA = sc.textFile(filename)
col_names = ['site', 'year', 'depth', 'coral_cover', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
#split the rdd into csv format
rdd = rddA.map(lambda row: row.split(','))

def convert_to_float(row):
    row[2] = int(row[2])
    for i in range(4, len(row)):
        row[i] = float(row[i])
    return row
rdd = rdd.map(convert_to_float)

def convert_to_coral_cover_percent(row):

    bleach_conversions = {
            "0": 0.0,
            "1L":0.05,
            "1U":0.10,
            "2L":0.20,
            "2U":0.30,
            "3L":0.40,
            "3U":0.50,
            "4L":0.625,
            "4U":0.75,
            "5L":0.875,
            "5U":1.0
        }
    row[3] = bleach_conversions[row[3][:2]]
    return row

rdd = rdd.map(convert_to_coral_cover_percent)


transposed_rdd = rdd.flatMap(lambda row: [(i, str(v)) for i,v in enumerate(row)])\
                    .reduceByKey(lambda a,b: a+","+b )\
                    .map(lambda row: (col_names[row[0]], np.asarray([float(i) for i in row[1].split(',')])))\

rdd_print(transposed_rdd,16)

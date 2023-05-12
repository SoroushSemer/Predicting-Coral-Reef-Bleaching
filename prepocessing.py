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

#split the rdd into csv format
rddA = rddA.map(lambda row: row.split(','))
print(rddA.count())

#get the column names
col_names = rddA.first()
pprint(col_names)
#filter out the header
rdd = rddA.filter(lambda row: row != col_names)


# #convert the row into dictionary format
# def col_format(row):
#     obj = {}
#     for i in col_names:
#         obj[i] = row[col_names.index(i)]
#     return obj
# rddA = rdd.map(col_format)

# def month_avgs(row):
#     start = col_names.index('1997-01')
#     end = col_names.index('2021-12')
#     avgs = [0]*12
#     count = [0]*12
#     for month in range(start, end+1):
#         if row[month] != '':
#             avgs[(month-start)%12] += float(row[month])
#             count[(month-start)%12] += 1
#     avgs = [avg/count[i] for i, avg in enumerate(avgs)]
#     variances = [0]*12
#     for month in range(start, end+1):
#         if row[month] != '':
#             variances[(month-start)%12] += (float(row[month]) - avgs[month%12])**2
#     variances = [variance/count[i] for i, variance in enumerate(variances)]

#     # [row.append(avg) for avg in avgs]
#     row = [(avgs[i], variances[i], count[i]) for i in range(12)]
#     return row
# stats = rdd.map(month_avgs)

# rdd_print(stats, 10)
'''Category Cover estimate
0 	0% 
1- 	>0-5% 
1+	>5-10% 
2- 	>10-20% 
2+ 	>20-30% 
3- 	>30-40% 
3+ 	>40-50% 
4- 	>50-62.5% 
4+ 	>62.5-75% 
5- 	>75-87.5% 
5+ 	>87.5-100%'''
def split_rows_by_year_convert_bleach(row):
    
    
    start = col_names.index('1997')
    end = col_names.index('2021')
    
    num_years = end - start+1

    rows = [[] for i in range(start, end+1)]

    for year in range(start, end+1):
        rows[year-start].append(row[0])
        rows[year-start].append(col_names[year])
        rows[year-start].append(row[col_names.index('depth')])
        rows[year-start].append(row[year])

    start = col_names.index('1997-01')
    end = col_names.index('2021-12')

    for month in range(start, end+1):
        rows[(month-start)//12].append(row[month])
    return rows
    
rdd = rdd.flatMap(split_rows_by_year_convert_bleach)
rdd = rdd.map(lambda row: ','.join(row))

# rdd.saveAsTextFile('output.csv')
f = open('output.csv', 'w')
for line in rdd.collect():
    f.write(line+"\n")
# rdd_print(rdd, 50)
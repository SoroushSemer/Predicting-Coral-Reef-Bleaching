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
from tabulate import tabulate
random.seed(0)

import matplotlib.pyplot as plt


sc = SparkContext()
sc.setLogLevel("ERROR")

def rdd_print(rdd, val:int):
    vals = rdd.take(val)
    for i in vals:
        print(i)

#hypothesis testing significant months

filename = 'temperature_reef_by_month_1_9\output.csv' #the data file to read into a stream

#read the file into the rdd
rddA = sc.textFile(filename)
col_names = ['site', 'year', 'depth', 'coral_cover', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
#split the rdd into csv format
rdd = rddA.map(lambda row: row.split(','))

def convert_to_float(row):
    row[2] = float(row[2])
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

only_data = rdd.map(lambda row:  row[2:])

transposed_rdd = only_data.flatMap(lambda row: [(i, str(v)) for i,v in enumerate(row)])\
                    .reduceByKey(lambda a,b: a+","+b )\
                    .map(lambda row: (col_names[row[0]+2], np.asarray([float(i) for i in row[1].split(',')])))

coral_cover = transposed_rdd.filter(lambda row: row[0] == 'coral_cover').collect()[0][1]
coral_cover = sc.broadcast(coral_cover)

def pearson(x, y):
    xmean = np.mean(x)
    ymean = np.mean(y)
    
    denom = math.sqrt(sum([(xi - xmean)**2 for xi in x])*sum([(yi - ymean)**2 for yi in y])) 
    if denom == 0:
        return 0

    numer = sum([(x[i]-xmean)*(y[i]-ymean) for i in range(len(x))])

    return numer / denom

pearson_corr = transposed_rdd.map(lambda row: (row[0], pearson(row[1], coral_cover.value), row[1]))

n = rdd.count()
df = n - 2
df = sc.broadcast(df)
def p_val(pearson_corr):

    sqrt_df = math.sqrt(df.value)
    t_val_numer = pearson_corr * sqrt_df
    t_val_denom = math.sqrt(1 - pearson_corr**2)
    t_val = t_val_numer / t_val_denom

    p_val = 2* tdist.sf(abs(t_val), df.value)
    return p_val

associations = pearson_corr.map(lambda row: (row[0], row[1], p_val(row[1]), row[2]))

print(associations.first())


associations = associations.filter(lambda row: row[0] != 'coral_cover').sortBy(lambda row: abs(row[1]))
associations = associations.collect()

print("Associations with coral cover")
print("factor\t\tcorrelation\t\tp-value")
print("------\t\t-----------\t\t------------------------")

for i in associations:
    print(i[0],"\t\t",round(i[1],7),"\t\t",round(i[2],10))

# rdd_print(associations,16)



#hypothesis testing bleach thresholds

filename = 'temperature_reef_by_month_1_9\All_Depths copy.csv' #the data file to read into a stream

#read the file into the rdd
rdd = sc.textFile(filename)

#split the rdd into csv format
rdd = rdd.map(lambda row: row.split(','))
print(rddA.count())

#get the column names
col_names = rdd.first()
#filter out the header
rdd = rdd.filter(lambda row: row != col_names)

def get_mean_n(row):
    start = col_names.index('1997-01')
    end = col_names.index('2021-12')
    total = 0
    n = 0
    for i in range(start, end+1):
        if(row[i] != ''):
            total += float(row[i])
            n += 1
    mean = total / n
    row.append(mean)
    row.append(n)

    return row
rdd_with_mean_n = rdd.map(get_mean_n)

def get_stdev(row):
    start = col_names.index('1997-01')
    end = col_names.index('2021-12')
    mean = row[-2]
    total = 0
    for i in range(start, end+1):
        if(row[i] != ''):
            total += (float(row[i]) - mean)**2

    stdev = math.sqrt(total / (row[-1]-1))
    row.append(stdev)
    return row

rdd_with_mean_n_stdev = rdd_with_mean_n.map(get_stdev)

# bleach_thresholds = {
#     1: 30.0,
#     2: 27.7,
#     3: 27.5,
#     4: 28.8,
#     5: 0.40,
#     6: 0.50,
#     7: 0.625,
#     8: 0.75,
#     9: 0.875,
# }

bleach_threshold = 28.6111

def get_t_val(row):
    t = (row[-3] - bleach_threshold) / (row[-1] / math.sqrt(row[-2]))
    row.append(t)
    return row

rdd_with_t = rdd_with_mean_n_stdev.map(get_t_val)

def get_p_val(row):
    p = tdist.sf(row[-1], row[-3])
    row.append(p)
    return row

rdd_with_p = rdd_with_t.map(get_p_val)

results = rdd_with_p.sortBy(lambda row: abs(row[-1]))\
                    .map(lambda row: [row[0],row[col_names.index('REEF_NAME')],row[col_names.index('depth')],str(row[-1])])

results = results.collect()

def convert_bleach_thresholds(row):
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
    start = col_names.index('1997')
    end = col_names.index('2021')
    for i in range(start, end+1):
        if(row[i] != ''):
            row[i] = bleach_conversions[row[i][:2]]
    return row

at_danger = rdd_with_p.filter(lambda row: row[-1]<0.05)\
                        .map(convert_bleach_thresholds)\
                        .map(lambda row: [row[0],row[col_names.index('REEF_NAME')],row[col_names.index('depth')],[float(row[i]) for i in range(col_names.index('1997-01'), col_names.index('2021-12')+1)], row[col_names.index('1997'):col_names.index('2021')+1]])\
                        .collect()


x = col_names[col_names.index('1997-01'):col_names.index('2021-12')+1]
y = [bleach_threshold for i in range(len(x))]
plt.plot(x,y, label='Bleaching Threshold', linestyle='--')

for row in at_danger:
    plt.plot(x, row[3], label=row[0]+" "+row[1] +" @"+str(round(float(row[2]),1))+"m")

plt.xticks([str(i)+"-01" for i in range(1997,2022,2)])
plt.ylabel('Water Temperature (°C)')
plt.xlabel('Month')
plt.legend()
plt.show()

x = col_names[col_names.index('1997'):col_names.index('2021')+1]
for row in at_danger:
    plt.plot(x, row[4], label=row[0]+" "+row[1] +" @"+str(round(float(row[2]),1))+"m")

plt.xticks([str(i) for i in range(1997,2022,2)])
plt.ylabel('Coral Cover')
plt.xlabel('Year')
plt.legend()
plt.show()

print("Reefs significantly above bleach threshold")
print(tabulate(results, headers=['Site','Reef','Depth','p-value']))
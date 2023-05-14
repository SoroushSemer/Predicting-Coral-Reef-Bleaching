import pandas as pd
import os
from haversine import haversine, Unit

f_in = "depth_1_9_average_group"

f_reef = "reef loc.csv"

for folder_name in os.listdir(f_in):
    # num = 0
    print(folder_name)

    folder_name = os.path.join(f_in, folder_name)

    for file_name in os.listdir(folder_name):
        # num = num+1

        file_name = os.path.join(folder_name, file_name)

        # print(file_name)
        # print(output_filename)
        station_csv = pd.read_csv(file_name)
        reef_csv = pd.read_csv(f_reef)

        station = station_csv.head(1)
        station_coord = (float(station.iat[0,6]),float(station.iat[0,7])) #lat, lon

        closest_reef = None
        closest_reef_dist = float('inf')

        for i, row in reef_csv.iterrows():
            # print(row)
            reef_coord = (row['LATITUDE'],row['LONGITUDE'])
            haversine_dist = haversine(station_coord,reef_coord, unit='mi')
            if(haversine_dist<closest_reef_dist):
                closest_reef_dist = haversine_dist
                closest_reef = row

        station_csv['REEF_NAME'] = closest_reef['REEF_NAME']
        station_csv['REEF_ID'] = closest_reef['REEF_ID']
        station_csv['REEF_lat'] = closest_reef['LATITUDE']
        station_csv['REEF_lon'] = closest_reef['LONGITUDE']

        station_csv.to_csv(file_name, mode='w', index=False)

        # print(file_name)
    # print(f"{folder_name}: {num}")

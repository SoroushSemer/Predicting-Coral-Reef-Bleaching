import pandas as pd
import os

def get_year_array(latest_year, num):
    month_list = []
    for year in range(latest_year-num+1,latest_year+1):
        # month_list.append(year)
        for month in range(1,13):
            month_list.append(f"{year}-{month:02}")
    return month_list

# def get_year_array(latest_year, num):
#     date_list = []
#     for year in range(latest_year-num+1,latest_year+1):
#         # month_list.append(year)
#         for month in range(1,13):
#             if(month==2):
#                 if(year%4==0):
#                     for date in range(1,30):
#                         date_list.append(f"{year}-{month:02}-{date:02}")
#                 else:
#                     for date in range(1,29):
#                         date_list.append(f"{year}-{month:02}-{date:02}")
#             else:
#                 if(month<=7):
#                     if(month%2==1):
#                         for date in range(1,32):
#                             date_list.append(f"{year}-{month:02}-{date:02}")
#                     else:
#                         for date in range(1,31):
#                             date_list.append(f"{year}-{month:02}-{date:02}")
#                 else:
#                     if(month%2==0):
#                         for date in range(1,32):
#                             date_list.append(f"{year}-{month:02}-{date:02}")
#                     else:
#                         for date in range(1,31):
#                             date_list.append(f"{year}-{month:02}-{date:02}")
#     return date_list

f_in = "depth_1_9_average_group_c"
f_out = "depth_1_9_average_group_f"

for folder_name in os.listdir(f_in):
    # num = 0
    print(folder_name)
    out_file_name = os.path.join(f_out, folder_name)+".csv"

    folder_name = os.path.join(f_in, folder_name)

    # print(out_file_name)
    # print(folder_name)
    # print()

    for file_name in os.listdir(folder_name):
        # num = num+1
        file_name = os.path.join(folder_name, file_name)
        
        if(os.path.isfile(out_file_name)):
            add_header = False
        else:
            add_header = True

        # print(file_name)
        # print(output_filename)
        csv = pd.read_csv(file_name)

        depth_with_month = csv.head(1).copy(deep=True)
        depth_with_month = depth_with_month.drop(columns=['time', 'cal_val', 'qc_val', 'qc_flag'])

        month_arr = get_year_array(2021,25)

        column_df = pd.DataFrame(columns=month_arr)

        depth_with_month = pd.concat((depth_with_month,column_df),axis=1)

        for i, row in csv.iterrows():
            if(row['time'] in depth_with_month):
                depth_with_month[row['time']] = row['qc_val']

        depth_with_month.to_csv(out_file_name, mode='a', index=False, header=add_header)

        # csv = csv.groupby(['site', 'site_id', 'subsite', 'subsite_id', 'depth', 'parameter', 'lat', 'lon', 'gbrmpa_reef_id', 'time'])

        # print(file_name)
    # print(f"{folder_name}: {num}")

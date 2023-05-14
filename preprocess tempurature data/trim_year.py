import pandas as pd
import os

f_in = "depth_1_9_average_group_c"

for folder_name in os.listdir(f_in):
    # num = 0
    print(folder_name)

    folder_name = os.path.join(f_in, folder_name)

    for file_name in os.listdir(folder_name):
        # num = num+1

        file_name = os.path.join(folder_name, file_name)

        # print(file_name)
        # print(output_filename)
        csv = pd.read_csv(file_name)
        if(csv.shape[0]<48):
            os.remove(file_name)

        # print(file_name)
    # print(f"{folder_name}: {num}")

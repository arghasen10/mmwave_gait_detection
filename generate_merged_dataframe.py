import numpy as np
import pandas as pd
import torch
import glob
from datetime import datetime, timedelta
import glob
from datetime import datetime, timedelta
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
from helper import *

annotated_df = pd.read_csv('annotation.csv')
annotated_df['datetime'] = pd.to_datetime(annotated_df['Datetime'])
annotated_df.drop(columns=['Datetime'], inplace=True)


def process_mmwave_data(filename):
    start_time = '-'.join(filename.split('/')[-1].split('.')[0].split('_')[3:][:-1])[:-1]
    start_time_object = datetime.strptime(start_time, "%Y-%m-%d-%H-%M-%S")
    print(f"Start Time: {start_time_object}")

    datetime_list = []
    curr_datetime = start_time_object
    
    with open(filename, 'rb') as f:
        adc_data = np.load(f)

    for row in adc_data:
        curr_datetime += timedelta(milliseconds=100)
        curr_datetime_str = datetime.strftime(curr_datetime,"%Y-%m-%d %H:%M:%S.%f")
        datetime_list.append(curr_datetime_str)

    print(f"Generated {len(datetime_list)} timestamps")
    return adc_data, datetime_list

mmwave_files = glob.glob('dataset/raw_data_day1/*.npy')

data_rows = []

for file in mmwave_files:
    adc_data, datetime_list = process_mmwave_data(file)

    time_index = 0  
    for radar_cube in adc_data:
        radar_cube = np.apply_along_axis(DCA1000.organize, 0, radar_cube, num_chirps=182*3, num_rx=4, num_samples=256)
        radar_cube = dsp.range_processing(radar_cube)
        mean = radar_cube.mean(0)
        radar_cube = radar_cube - mean
        radar_cube = np.concatenate((radar_cube[0::3, ...], radar_cube[1::3, ...], radar_cube[2::3, ...]), axis=1)
        radar_cube = radar_cube.reshape(182, 3, 4, 256)
        
        doppler_fft = dopplerFFT(radar_cube)
        
        pcds = frame2pointcloud(doppler_fft)

        rangeOutput = np.transpose(np.absolute(radar_cube), (1, 2, 0, 3)).sum(axis=(0, 1, 2))

        rangeDoppler = np.absolute(doppler_fft).sum(axis=(0, 1))

        feature3 = pcds

        feature_dict = {
            'datetime': datetime_list[time_index],
            'Feature_1': rangeOutput.tolist(), 
            'Feature_2': rangeDoppler.tolist(),  
            'Feature_3': feature3.tolist()  
        }

        data_rows.append(feature_dict)

        time_index += 1
    print("File completed: "+file)

df = pd.DataFrame(data_rows)
print(df.head())


df['datetime'] = pd.to_datetime(df['datetime'])
df.columns = [['datetime', 'range', 'rangedoppler', 'pointcloud']]
df = df[['datetime', 'range', 'rangedoppler']]
mmdf=pd.DataFrame({'datetime':df['datetime'].apply(lambda e: pd.to_datetime(str(e[0])),axis=1),'range':df['range'].apply(lambda e: e[0],axis=1),'rangedoppler':df['rangedoppler'].apply(lambda e: e[0],axis=1)})

mmdf_sorted=mmdf.sort_values(by='datetime')

annotated_df['time']=annotated_df['datetime'].apply(lambda e: pd.to_datetime(str(e)))
merged_df = pd.merge_asof(mmdf_sorted,annotated_df,on='datetime',direction='nearest')
merged_df.to_pickle('merged_df.pkl')
import numpy  as np
import glob
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000


mmwave_files = glob.glob('dataset/raw_data_day1/*.npy')


for file in mmwave_files:
    with open(file, 'rb') as f:
        data = np.load(f)
        timeArr = np.load(f)
        for radar_cube in data: 
            radar_cube = np.apply_along_axis(DCA1000.organize, 0, radar_cube, num_chirps=182*3, num_rx=4, num_samples=256)
            radar_cube = dsp.range_processing(radar_cube)
            mean = radar_cube.mean(0)                 
            radar_cube = radar_cube - mean
            radar_cube = np.concatenate((radar_cube[0::3,...], radar_cube[1::3,...], radar_cube[2::3,...]), axis=1)
            print(radar_cube.shape)
        print(data.shape)
        print(timeArr[1])
    break


# with open('')
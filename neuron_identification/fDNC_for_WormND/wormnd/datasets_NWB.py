import os
import sys
import h5py
import pickle
import numpy as np 
import pandas as pd
 
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt 
from pynwb import load_namespaces, get_class, register_class, NWBFile, TimeSeries, NWBHDF5IO
 
from __init__ import * 
from utils import * 
 
DATASETS_ALL = ['EY', 'HL', 'KK', 'NP', 'SF', 'SK1', 'SK2']
 
Dandi_IDS = {
    'EY': '000541',
    'HL': '000714',
    'KK': '000692',
    'NP': '000715',
    'SF': '000776',
    'SK1': '000565',
    'SK2': '000472'
}
 
MAX_IMG_VALUE = {
    'EY': 4095,
    'HL': 65535,
    'KK': 65535,
    'NP': 65535,
    'SF': np.nan,
    'SK1': 4095,
    'SK2': 4095,
}
## TODO: the data summary  
## TODO: SF is not available


class NWB_data():
    def __init__(self, filepath):
        with NWBHDF5IO(filepath, mode='r', load_namespaces=True) as io:
            read_nwb = io.read()
            self.identifier = read_nwb.identifier
            self.seg = read_nwb.processing['NeuroPAL']['NeuroPALSegmentation']['NeuroPALNeurons'].voxel_mask[:] 

            self.labels = read_nwb.processing['NeuroPAL']['NeuroPALSegmentation']['NeuroPALNeurons']['ID_labels'][:]
            self.channels = read_nwb.acquisition['NeuroPALImageRaw'].RGBW_channels[
                            :]  # get which channels of the image correspond to which RGBW pseudocolors
            self.image = read_nwb.acquisition['NeuroPALImageRaw'].data[:]
            self.scale = read_nwb.imaging_planes['NeuroPALImVol'].grid_spacing[
                         :]  # get which channels of the image correspond to which RGBW pseudocolors

    def preprocess(self, channel_number=3):
        print(self.identifier)
        self.labels = ["".join(label) for label in self.labels]

        blobs = pd.DataFrame.from_records(self.seg, columns=['X', 'Y', 'Z', 'weight'])
        blobs = blobs.drop(['weight'], axis=1)

        if len(self.channels) == 4 and channel_number == 3:
            RGB_channels = self.channels[:-1]
        else:
            RGB_channels = self.channels

        RGB = self.image[:, :, :, RGB_channels]

        blobs = blobs[(blobs['x'] < RGB.shape[0]) & (blobs['y'] < RGB.shape[1]) & (blobs['z'] < RGB.shape[2])]

        idx_keep = [i for i, row in blobs.iterrows() if
                    (row['x'] < RGB.shape[0]) and (row['y'] < RGB.shape[1]) and (row['z'] < RGB.shape[2])]
        if channel_number == 3:
            blobs[['R', 'G', 'B']] = [RGB[row['x'], row['y'], row['z'], :] for i, row in blobs.iterrows()]
        elif channel_number == 4:
            blobs[['R', 'G', 'B', 'W']] = [RGB[row['x'], row['y'], row['z'], :] for i, row in blobs.iterrows()]

        blobs[['xr', 'yr', 'zr']] = [[row['x'] * self.scale[0], row['y'] * self.scale[1], row['z'] * self.scale[2]]
                                     for i, row in
                                     blobs.iterrows()]
        blobs['ID'] = [self.labels[i] for i in idx_keep]

        self.blobs = blobs.replace('nan', '', regex=True)
        # blobs = blobs[blobs['ID'].isin(atlas_neurons)]
        self.RGB = RGB
        return self.blobs, self.RGB



def transform_NWB_data_into_labels(data_NWB, need_norm=True, channel_number=4):
    # data_NWB = NWB_data(file_path)
    identifier = data_NWB.identifier
    if channel_number==3:
        blobs, RGB = data_NWB.preprocess(channel_number=3)
        neuron_colors = blobs[['R', 'G', 'B']].values
    elif channel_number==4:
        blobs, RGB = data_NWB.preprocess(channel_number=4)
        # print(len(blobs))
        # Create temp_color matrix
        neuron_colors = blobs[['R', 'G', 'B', 'W']].values


    # # Mapping ID to numeric values
    # id_mapping = {id_val: idx for idx, id_val in enumerate(blobs['ID'].unique())}
    # blobs['ID_numeric'] = blobs['ID'].map(id_mapping)
    neuron_labels = blobs['ID']

    # Normalize x y z for temp_pos 
    if need_norm:
        # blobs['x_norm'] = min_max_normalize(blobs['xr'])
        # blobs['y_norm'] = min_max_normalize(blobs['yr'])
        # blobs['z_norm'] = min_max_normalize(blobs['zr'])
        blobs['x_norm'] = center_sccale_normalize(blobs['xr'])
        blobs['y_norm'] = center_sccale_normalize(blobs['yr'])
        blobs['z_norm'] = center_sccale_normalize(blobs['zr'])
        # Create temp_pos matrix
        neuron_positions = blobs[['x_norm', 'y_norm', 'z_norm']].values
    else:
        neuron_positions = blobs[['xr', 'yr', 'zr']].values
        # neuron_positions = blobs[['x', 'y', 'z']].values
    return neuron_colors, neuron_positions, neuron_labels



def convert_nwb_data_for_fDNC(path_from=PATH_DATA, path_to=PATH_DATA_PKL, need_normalization=False):
    import warnings
    warnings.filterwarnings("ignore")

    acc_all = []
    combined_data_all = {}
    for dataset in DATASETS_ALL:
        dandi_id = Dandi_IDS[dataset]
        print(dataset, dandi_id) 
        path_from_1 = os.path.join(path_from, dandi_id) 
        file_path_all = get_all_files(path_from_1)
        file_path_all = [file_path for file_path in file_path_all if file_path.endswith('.nwb')]
        # print(file_path_all)
        data_all = {}

        for data_id in range(len(file_path_all)):
            file_path = file_path_all[data_id]
            data_NWB = NWB_data(file_path)
            temp_color, temp_pos, temp_label = transform_NWB_data_into_labels(data_NWB, need_norm=need_normalization)
            data_all[data_NWB.identifier] = [temp_color, temp_pos, temp_label]
        combined_data_all[dataset] = data_all
        if need_normalization: 
            name = os.path.join(path_to, f'pos_col_labl__for_{dataset}.pkl')
        else: 
            name = os.path.join(path_to, f'pos_col_labl__for_{dataset} (wo_norm) pixel.pkl')
        with open(name, "wb") as f:
            pickle.dump(data_all, f, protocol=pickle.HIGHEST_PROTOCOL)

    return combined_data_all



def main():      
    # defined the PATH_DATA and PATH_MODEL and PATH_RESULT in the __init__.py
    path_from=PATH_DATA
    path_to=PATH_DATA_PKL
    need_normalization=False
    
    combined_data_all = convert_nwb_data_for_fDNC(path_from, path_to, need_normalization)
    print(combined_data_all.keys())


if __name__ == '__main__':
    main()
 
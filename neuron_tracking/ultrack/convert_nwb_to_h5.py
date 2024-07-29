# coding: utf-8

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

from pynwb import NWBHDF5IO
from glob import glob
from joblib import Parallel, delayed


def process_file(path, data_directory):
    with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
        read_nwb = io.read()

        if "CalciumImageSeries" in read_nwb.acquisition:
            calcium_activity = read_nwb.processing["CalciumActivity"]["CalciumSeriesSegmentation"]
            if '000776' in path:
                plane_segmentations = calcium_activity.plane_segmentations['Aligned_neuron_coordinates']
                plane_segmentation_dicts = {t: seg for t, seg in enumerate(plane_segmentations)}
            elif hasattr(calcium_activity, "plane_segmentations"): # 472, 541, 565
                plane_segmentation_dicts = calcium_activity.plane_segmentations
            elif '000692' in path:
                plane_segmentations = np.zeros((len(calcium_activity.data), 250, 3), dtype=np.uint16)
                for k, v in enumerate(calcium_activity.data):
                    W, H, D = v.shape
                    for idx in range(v.max()):
                        coordinates = np.argwhere(v == idx + 1)
                        if len(coordinates) == 0:
                            plane_segmentations[k][idx] = plane_segmentations[k - 1][idx]
                        else:
                            plane_segmentations[k][idx] = coordinates.mean(axis=0).round()
                plane_segmentations = plane_segmentations[:, plane_segmentations.max(axis=(0, 2)) > 0]
                plane_segmentation_dicts = {t: seg for t, seg in enumerate(plane_segmentations)}
            else:
                print('Error with', path)
                return

            seg_data = [None] * len(plane_segmentation_dicts)
            labels_data = []
            for seg_tpoint, segmentation in plane_segmentation_dicts.items():
                if isinstance(seg_tpoint, str):
                    seg_tpoint = int(seg_tpoint.split('_')[-1])
                if isinstance(segmentation, np.ndarray):
                    voxel_masks = segmentation
                else:
                    voxel_masks = segmentation["voxel_mask"][:]
                if isinstance(voxel_masks, np.ndarray):
                    voxel_masks = voxel_masks.squeeze()
                seg_list = [np.array(voxel_mask.tolist()).flatten()[:3] 
                            for voxel_mask in voxel_masks]
                seg_data[seg_tpoint] = seg_list

            seg_data = np.array(seg_data)

            dataset = read_nwb.acquisition["CalciumImageSeries"].data
            calcium_data = np.empty(dataset.shape[:4], dtype=np.uint8)
            for i in range(dataset.shape[0]):
                if len(dataset.shape) == 4:  # single channel
                    slice_data = dataset[i]
                else:
                    slice_data = dataset[i, :, :, :, 0]
                lower_bound = np.percentile(slice_data, 5)
                upper_bound = np.percentile(slice_data, 99)
                slice_data = np.clip(slice_data, lower_bound, upper_bound)

                calcium_data[i] = ((slice_data - lower_bound)* 255 / (upper_bound - lower_bound)).round().astype(np.uint8)

            # Define the path for the new H5 file based on the original NWB file name
            h5_filename = f"{data_directory}/calcium_h5/{path.split('/')[-1]}/{path.split('/')[-1].replace('.nwb', '.h5')}"
            with h5py.File(h5_filename, 'w') as h5file:
                # Create datasets in the new H5 file
                h5file.create_dataset('calcium_image', data=calcium_data, compression="gzip")
                h5file.create_dataset('calcium_segmentation', data=seg_data, compression=None)

            print('finished', path)



def main(nwb_directory):
    nwb_filenames = glob(f'{nwb_directory}/*/*/*.nwb')
    print(len(nwb_filenames))

    # Parallel processing
    Parallel(n_jobs=3)(delayed(process_file)(path, nwb_directory) for path in nwb_filenames);

    # For loop processing
    # for path in nwb_filenames:
    #     process_file(path, nwb_directory)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ultrack with pretrained stardist weights.")
    parser.add_argument('nwb_directory', type=str, help='Path to the data directory')
    args = parser.parse_args()
    
    main(args.nwb_directory)







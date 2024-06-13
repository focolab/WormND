from pynwb import NWBHDF5IO
import pandas as pd
import tifffile as tiff
import numpy as np

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
            
            
    def preprocess(self):
        print(self.identifier)
        self.labels = ["".join(label) for label in self.labels]

        blobs = pd.DataFrame.from_records(self.seg, columns=['X', 'Y', 'Z', 'weight'])
        blobs = blobs.drop(['weight'], axis=1)

        RGB_channels = self.channels[:-1]

        RGB = self.image[:, :, :, RGB_channels]

        blobs = blobs[(blobs['x'] < RGB.shape[0]) & (blobs['y'] < RGB.shape[1]) & (blobs['z'] < RGB.shape[2])]

        idx_keep = [i for i, row in blobs.iterrows() if
                    (row['x'] < RGB.shape[0]) and (row['y'] < RGB.shape[1]) and (row['z'] < RGB.shape[2])]

        blobs[['R', 'G', 'B']] = [RGB[row['x'], row['y'], row['z'], :] for i, row in blobs.iterrows()]
        blobs[['xr', 'yr', 'zr']] = [[row['x'] * self.scale[0], row['y'] * self.scale[1], row['z'] * self.scale[2]] for i, row in
                                     blobs.iterrows()]
        blobs['ID'] = [self.labels[i] for i in idx_keep]

        self.blobs = blobs.replace('nan', '', regex=True)
        self.RGB = RGB

        self.RGB = (self.RGB - self.RGB.min()) / (self.RGB.max() - self.RGB.min())
        self.RGB = self.RGB * 255
        self.RGB = self.RGB.astype('uint16')

        return self.blobs, self.RGB
    

    def to_tiff(self, output_filepath):
        RGB_reordered = np.transpose(self.RGB, (2, 3, 0, 1))
        tiff.imwrite(output_filepath, RGB_reordered, photometric='minisblack')
        return RGB_reordered
    

    def to_gray_scale(self, output_filepath):
        RGB = (self.RGB - self.RGB.min()) / (self.RGB.max() - self.RGB.min())
        gray_images = 0.299 * RGB[:, :, :, 0] + 0.587 * RGB[:, :, :, 1] + 0.114 * RGB[:, :, :, 2] 
        gray_images = gray_images * 255
        gray_images = gray_images.astype('uint16')
        tiff.imwrite(output_filepath, gray_images, photometric='minisblack')
        return gray_images
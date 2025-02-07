import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io, core, utils, denoise
from cellpose.io import imread
import os
import sys
import time
import cv2

def visualize(image, mask, saved_path):
    z, c, w, h = image.shape
    fig, axes = plt.subplots(nrows=int(np.ceil(z / 7)), ncols=7)
    axes = axes.flatten()

    for i in range(z):
        img_slice = np.transpose(image[i], (1, 2, 0))
        axes[i].imshow(img_slice)
        outlines = utils.outlines_list(mask[i])

        for o in outlines:
            axes[i].plot(o[:, 0], o[:, 1], color=[1, 1, 0], linewidth=0.5)

        axes[i].axis('off')

    for j in range(z, len(axes)):
        fig.delaxes(axes[j])
    
    plt.savefig(saved_path, dpi=300)
    plt.close()

def cell_diameter_in_pixels(voxel_size, cell_diameter_um):
    average_voxel_size = (voxel_size[0] + voxel_size[1] + voxel_size[2]) / 3
    cell_diameter_pixels = cell_diameter_um / average_voxel_size
    return cell_diameter_pixels 

def detect_anomaly(masks, threshold=1000):
    for mask in masks:
        n_instances = np.unique(mask).shape[0]
        if n_instances > threshold:
            print(n_instances)
            return True
    return False

def write_log_to_file(filename, message):
    print(message)
    with open(filename, 'a') as file:
        file.write(message + '\n')

if __name__ == "__main__":
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    sys.path.append(project_dir)
    from NWB_data import NWB_data

    sessions = ['000541','000472','000692','000715', '000714','000776', '000565']
    log_file = '/scratch/th3129/wormID/results/cellpose/inference.log'

    for session in sessions:
        start_time = time.time()

        train_input_folder_path = f'/scratch/th3129/wormID/datasets/tiff_files/{session}/train'
        test_input_folder_path = f'/scratch/th3129/wormID/datasets/tiff_files/{session}/test'
        label_folder_path = f'/scratch/th3129/wormID/datasets/{session}'
        output_folder_path = f"/scratch/th3129/wormID/results/cellpose/{session}"

        use_GPU = core.use_gpu()
        io.logger_setup()

        train_files = [os.path.join(train_input_folder_path, file) for file in os.listdir(train_input_folder_path) if file.endswith("_img.tiff")]
        test_files = [os.path.join(test_input_folder_path, file) for file in os.listdir(test_input_folder_path) if file.endswith("_img.tiff")]
        files = train_files + test_files

        subdir_path = os.path.join(label_folder_path, os.listdir(label_folder_path)[0])
        label_file_path = os.path.join(subdir_path, os.listdir(subdir_path)[0])
        nwb = NWB_data(label_file_path)
        print(session, nwb.scale)

        diameter = cell_diameter_in_pixels(nwb.scale, cell_diameter_um=10)
        print(nwb.scale, diameter, int(diameter))
        
        imgs = [imread(f) for f in files]
        nimg = len(imgs)

        # you can specify cell diameter and channels for your own dataset
        model = models.Cellpose(gpu=True, model_type='cyto3')
        masks, flows, styles, diams = model.eval(imgs, diameter=int(diameter), channels=[3, 1], do_3D=True, anisotropy=nwb.scale[2]/nwb.scale[0])
        
        if detect_anomaly(masks):
            print("WARNINGS: too many segmented instances detected, fallback to default parameters")
            masks, flows, styles, diams = model.eval(imgs, channels=[0, 0], do_3D=True)
        
        end_time = time.time()
        message = f"Execution Time: {end_time-start_time} seconds for {len(masks)} images"
        write_log_to_file(log_file, message)

        for i in range(len(imgs)):
            file = os.path.basename(files[i])
            saved_path = os.path.join(output_folder_path, os.path.splitext(file)[0]+'.png')
            # visualize(imgs[i], masks[i], saved_path)
            np.save(os.path.splitext(saved_path)[0]+'.npy', masks[i])
from NWB_data import NWB_data
import os
import numpy as np
from scipy.ndimage import distance_transform_edt
import tifffile as tiff
import matplotlib.pyplot as plt


def create_sphere_mask(volume_shape, centroid, radius=5, voxel_size=(5, 5, 5)):
    x0, y0, z0 = centroid
    vx, vy, vz = voxel_size
    x, y, z = np.indices(volume_shape)

    distance = np.sqrt(((x-x0)*vx)**2 + ((y-y0)*vy)**2 + ((z-z0)*vz)**2)   
    mask = distance <= radius
    return mask


def generate_mask(centroids, img, radius=5, voxel_size=(5, 5, 5)):
    num_centroids = len(centroids)
    labels = np.arange(1, num_centroids + 1)

    instance_masks = np.zeros_like(img, dtype=np.int32)

    for label, centroid in zip(labels, centroids):
        mask = create_sphere_mask(img.shape, centroid, radius=radius, voxel_size=voxel_size)
        instance_masks[mask] = label

    instance_masks = instance_masks.transpose(2, 0, 1)

    return instance_masks


def visualize_image_mask(image, gt_cell, path, scatter=False):
    fig, axs = plt.subplots(nrows=1, ncols=4)

    axs[0].imshow(np.max(image, axis=2))
    axs[1].imshow(np.max(image[:, :, :, 0], axis=2), cmap='Reds')
    axs[2].imshow(np.max(image[:, :, :, 1], axis=2), cmap='Greens')
    axs[3].imshow(np.max(image[:, :, :, 2], axis=2), cmap='Blues')

    if scatter:
        axs[0].scatter(gt_cell[:, 1], gt_cell[:, 0], s=2, alpha=0.5, c='white')
        axs[1].scatter(gt_cell[:, 1], gt_cell[:, 0], s=2, alpha=0.5, c='black')
        axs[2].scatter(gt_cell[:, 1], gt_cell[:, 0], s=2, alpha=0.5, c='black')
        axs[3].scatter(gt_cell[:, 1], gt_cell[:, 0], s=2, alpha=0.5, c='black')

    for ax in axs.flat:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.savefig(path, dpi=300)
    plt.close()


def visualize_image_points(image, gt_cell, path):
    fig, ax = plt.subplots()

    ax.imshow(np.transpose(np.max(image, axis=2), (1,0,2)))
    ax.scatter(gt_cell[:, 0], gt_cell[:, 1], s=2, c='white')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.savefig(path, dpi=300)
    plt.close()

if __name__ == "__main__": 
    sessions = ['000472', '000692', '000715']

    for session in sessions:
        input_folder_path = f"/scratch/th3129/wormID/datasets/{session}" 
        train_folder_path = f"/scratch/th3129/wormID/datasets/tiff_files/{session}/train" 
        test_folder_path = f"/scratch/th3129/wormID/datasets/tiff_files/{session}/test" 
        image_path = f"/scratch/th3129/wormID/datasets/tiff_files/{session}/gt" 
        
        raw_data, masks, files_path = [], [], [] 
        for subdir in os.listdir(input_folder_path): 
            subdir_path = os.path.join(input_folder_path, subdir) 

            if os.path.isdir(subdir_path): 
                for file in os.listdir(subdir_path): 

                    if file.endswith('.nwb'): 
                        input_file_path = os.path.join(subdir_path, file) 
                        nwb = NWB_data(input_file_path) 
                        blobs, RGB = nwb.preprocess() 
                        
                        centroids = blobs[['x', 'y', 'z']].to_numpy() 
                        mask = generate_mask(centroids, RGB[:, :, :, 0], radius=2.5, voxel_size=nwb.scale) 
                        visualize_image_mask(RGB, centroids, os.path.join(image_path, os.path.splitext(file)[0]+'.png'))
                        visualize_image_points(RGB, centroids, 'points.png')

                        raw_data.append(nwb) 
                        masks.append(mask) 
                        files_path.append(file)

        train_test_split = 0.8 
        n_sessions = len(masks)
        training_size = int(n_sessions * train_test_split)
        
        for i in range(training_size):
            output_image_path = os.path.join(train_folder_path, os.path.splitext(files_path[i])[0]+'_img.tiff')
            raw_data[i].to_tiff(output_image_path)

            output_label_path = os.path.join(train_folder_path, os.path.splitext(files_path[i])[0]+'_masks.tiff')
            tiff.imwrite(output_label_path, masks[i], photometric='minisblack')

        for i in range(training_size, n_sessions):
            output_image_path = os.path.join(test_folder_path, os.path.splitext(files_path[i])[0]+'_img.tiff')
            raw_data[i].to_tiff(output_image_path)

            output_label_path = os.path.join(test_folder_path, os.path.splitext(files_path[i])[0]+'_masks.tiff')
            tiff.imwrite(output_label_path, masks[i], photometric='minisblack')    
import os
import numpy as np
import tifffile as tiff
from scipy.ndimage import label
import sys

def generate_masks(src_dir, dst_dir, label_dir):
    labels = {}
    for subdir in os.listdir(label_dir):
        subdir_path = os.path.join(label_dir, subdir)
        
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                label_file_path = os.path.join(subdir_path, file)
                
                if file.endswith('.nwb'):
                    nwb = NWB_data(label_file_path)
                    blobs, RGB = nwb.preprocess()
                    gt_labels = blobs[['x', 'y', 'z']].to_numpy()
                    labels[os.path.splitext(file)[0]] = gt_labels
    
    for file in os.listdir(dst_dir):
        if file.endswith("_img.tiff"):
            src_path = os.path.splitext(file)[0]+'.npy'
            dst_path = os.path.splitext(file)[0][:-4]+'_pseudomasks.tiff'

            src = os.path.join(src_dir, src_path)
            dst = os.path.join(dst_dir, dst_path)

            mask = np.load(src)
            gt_labels = labels[os.path.splitext(file)[0][:-4]]
            mask = mask.transpose(1, 2, 0)

            filtered_mask = filter_instances_by_ground_truth(mask, gt_labels)
            filtered_mask = filtered_mask.transpose(2, 0, 1)
            tiff.imwrite(dst, filtered_mask)


def filter_instances_by_ground_truth(mask, gt_labels):
    labeled_mask, num_features = label(mask)
    
    keep_labels = np.zeros(num_features + 1, dtype=bool)
    
    for x, y, z in gt_labels:
        label_at_point = labeled_mask[x, y, z]
        if label_at_point > 0:
            keep_labels[label_at_point] = True
    
    filtered_mask = np.zeros_like(mask)
    for label_num in range(1, num_features + 1):
        if keep_labels[label_num]:
            filtered_mask[labeled_mask == label_num] = label_num
    
    return filtered_mask


if __name__ == "__main__":
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    sys.path.append(project_dir)
    from NWB_data import NWB_data

    session = '000541'

    train_dir = f"/scratch/th3129/wormID/datasets/tiff_files/{session}/train" 
    test_dir = f"/scratch/th3129/wormID/datasets/tiff_files/{session}/test" 
    label_dir = f"/scratch/th3129/wormID/datasets/{session}" 
    mask_path = f"/scratch/th3129/wormID/results/cellpose/{session}"

    generate_masks(mask_path, train_dir, label_dir)
    generate_masks(mask_path, test_dir, label_dir)
            
    


from scipy.spatial.distance import cdist
import numpy as np
import os
from scipy.ndimage import center_of_mass
import sys
import matplotlib.pyplot as plt
import pandas as pd


def get_dist_threshold(gt_cells, scales):
    scaled_gt_cells = gt_cells.copy()
    scaled_gt_cells = scaled_gt_cells * scales
    # print(scales, scaled_gt_cells)
    
    dist = cdist(scaled_gt_cells, scaled_gt_cells, metric='euclidean')
    np.fill_diagonal(dist, np.inf)

    min_dist = np.min(dist, axis=1)
    min_dist.sort()
    dist_threshold = np.mean(min_dist[:5]) / 2

    return dist_threshold

def evaluate(predicted_cell, gt_cell, threshold=15):
    """
    predicted_cell: n x 3
    gt_cell: m x 3

    If there is no ground-truth cell within a valid distance, the cell prediction is counted as an FP
    If there are one or more ground-truth cells within a valid distance, the cell prediction is counted as a TP.
    The remaining ground-truth cells that are not matched with any cell prediction are counted as FN.

    return precision, recall, f1 score
    """
    dist = cdist(predicted_cell, gt_cell, metric='euclidean')
    n_pred, n_gt = dist.shape
    assert(n_pred != 0 and n_gt != 0)
    bool_mask = (dist <= threshold)
    tp, fp = 0, 0

    for i in range(len(predicted_cell)):
        neighbors = bool_mask[i].nonzero()[0]

        if len(neighbors) == 0:
            fp += 1
        else:
            gt_idx = min(neighbors, key=lambda j: dist[i, j])
            tp += 1
            bool_mask[:, gt_idx] = False
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (n_gt + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return precision, recall, f1


def mask_to_centroids(mask):
    mask = np.transpose(mask, (1, 2, 0))
    labels = np.unique(mask)
    centroids = []

    for label in labels:
        if label == 0:
            continue
        centroids.append(center_of_mass(mask == label))

    return np.array(centroids)

def visualize_labels(image, mask, predicted_cell, gt_cell, results_path):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
    axs[0].imshow(np.max(image, axis=2))
    axs[0].scatter(gt_cell[:, 1], gt_cell[:, 0], s=1, c='white')

    axs[1].imshow(np.max(mask, axis=0))

    axs[2].imshow(np.max(image, axis=2))
    axs[2].scatter(predicted_cell[:, 1], predicted_cell[:, 0], s=1, c='white')

    for ax in axs.flat:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.savefig(results_path, dpi=500)
    plt.close()


if __name__ == "__main__":
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    sys.path.append(project_dir)
    from NWB_data import NWB_data

    sessions = ['000541','000472','000692','000715']
    
    for session in sessions:

        label_folder_path = f"/scratch/th3129/wormID/datasets/{session}"
        results_folder_path = f"/scratch/th3129/wormID/results/micro-sam/{session}"
        metrics = []

        for subdir in os.listdir(label_folder_path):
            subdir_path = os.path.join(label_folder_path, subdir)
            
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.endswith('.nwb'):
                        label_file_path = os.path.join(subdir_path, file)
                        mask_file_path = os.path.join(results_folder_path, os.path.splitext(file)[0]+'_img.npy')

                        nwb = NWB_data(label_file_path)
                        blobs, RGB = nwb.preprocess()
                        mask = np.load(mask_file_path)

                        gt_labels = blobs[['x', 'y', 'z']].to_numpy()
                        pred_labels = mask_to_centroids(mask)
                        # threshold = get_dist_threshold(gt_labels, nwb.scale)

                        precision, recall, f1 = evaluate(pred_labels * nwb.scale, gt_labels * nwb.scale, threshold=3)
                        print(precision, recall, f1)
                        metrics.append({'worm':os.path.splitext(file)[0], 'precision': precision, 'recall': recall, 'f1_score': f1})
                        save_path = os.path.join(results_folder_path, os.path.splitext(file)[0]+'_masks.png')
                        visualize_labels(RGB, mask, pred_labels, gt_labels, save_path)
        
        df = pd.DataFrame(metrics)
        df.to_csv(os.path.join(results_folder_path, 'performance_metrics_dist3.csv'), index=False)
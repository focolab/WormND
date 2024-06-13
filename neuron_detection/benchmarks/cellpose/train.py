from cellpose import io, models, train, core, utils
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

def visualize(image, mask, saved_path):
    z, c, w, h = image.shape
    fig, axes = plt.subplots(nrows=int(np.ceil(z / 7)), ncols=7)
    axes = axes.flatten()

    for i in range(z):
        axes[i].imshow(image[i])
        outlines = utils.outlines_list(mask[i])

        for o in outlines:
            axes[i].plot(o[:, 0], o[:, 1], color=[1, 1, 0], linewidth=0.5)

        axes[i].axis('off')

    for j in range(z, len(axes)):
        fig.delaxes(axes[j])
    
    plt.savefig(saved_path, dpi=300)
    plt.close()

def visualize_gt_mask(images, labels):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(np.max(images, axis=0), (1, 2, 0)))
    plt.title("Input Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.max(labels, axis=0))
    plt.title("Ground Truth Mask")
    
    plt.savefig('gt_mask.png')

def visualize_flows(flow):
    z, w, h, c = flow[0].shape
    flow_hsv = flow[0]
    cell_prob = flow[2]

    fig, axes = plt.subplots(nrows=int(np.ceil(z / 7)), ncols=7)
    axes = axes.flatten()

    for i in range(z):
        axes[i].imshow(flow_hsv[i])
        
    for j in range(z, len(axes)):
        fig.delaxes(axes[j])
    
    plt.savefig("flows.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(nrows=int(np.ceil(z / 7)), ncols=7)
    axes = axes.flatten()

    for i in range(z):
        caxs = axes[i].imshow(cell_prob[i])
        fig.colorbar(caxs, ax=axes[i])

    for j in range(z, len(axes)):
        fig.delaxes(axes[j])
    
    plt.savefig("cell_prob.png", dpi=300)
    plt.close()

def load_data(sessions):
    all_files, all_images_2d, all_images_3d, all_labels = [], [], [], []

    for session in sessions:
        train_dir = f"/scratch/th3129/wormID/datasets/tiff_files/{session}/train" 
        test_dir = f"/scratch/th3129/wormID/datasets/tiff_files/{session}/test" 

        train_files = [os.path.join(train_dir, file) for file in os.listdir(train_dir) if file.endswith("_img.tiff")]
        test_files = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if file.endswith("_img.tiff")]
        all_files += train_files + test_files

        output = io.load_train_test_data(train_dir, test_dir, image_filter="_img",
                                        mask_filter="_masks", look_one_level_down=False)
        images, labels, image_names, test_images, test_labels, image_names_test = output
        
        print(images[0].shape, labels[0].shape, test_images[0].shape, test_labels[0].shape)
        
        images_2d = [np.transpose(image, (0, 2, 3, 1))[z] for image in images for z in range(len(image))]
        images_3d = [np.transpose(image, (0, 2, 3, 1)) for image in images]
        visualize(images_3d[0], labels[0], "gt_mask.png")

        labels = [label[z] for label in labels for z in range(len(label))]
        
        test_images_2d = [np.transpose(test_image, (0, 2, 3, 1))[z] for test_image in test_images for z in range(len(test_image))]
        test_images_3d = [np.transpose(test_image, (0, 2, 3, 1)) for test_image in test_images]
        test_labels = [test_label[z] for test_label in test_labels for z in range(len(test_label))]
        
        print(images_2d[0].shape, labels[0].shape, test_images_2d[0].shape, test_labels[0].shape)
        
        all_images_2d += images_2d + test_images_2d
        all_images_3d += images_3d + test_images_3d
        all_labels += labels + test_labels
    
    return all_files, all_images_2d, all_images_3d, all_labels

def extract_parts(path):
    parts = path.split('/')
    part1 = parts[0]
    part2 = parts[2].split('.')[0]
    return part1, part2

def cross_validate(all_files, all_images_2d, all_images_3d, all_labels, i, split_file_path):
    split_file = pd.read_csv(split_file_path)
    split_file[['session', 'worm']] = split_file['filename'].apply(lambda x: pd.Series(extract_parts(x)))
    split_file['dataset_split'].astype(int)
    
    worms_in_split = split_file[split_file['dataset_split'] == i]['worm'].tolist()
    all_files = [path.split('/')[-1].split('.')[0][:-4] for path in all_files]
    train_images, train_labels, train_files, test_images_3d, test_labels, test_files = [], [], [], [], [], []
    
    for file, img_2d, img_3d, label in zip(all_files, all_images_2d, all_images_3d, all_labels):
        if file in worms_in_split:
            train_images.append(img_2d)
            train_labels.append(label)
            train_files.append(file)
        else:
            test_images_3d.append(img_3d)
            test_labels.append(label)
            test_files.append(file)
    
    return train_images, train_labels, train_files, test_images_3d, test_labels, test_files


if __name__ == "__main__":
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    sys.path.append(project_dir)
    from NWB_data import NWB_data

    use_GPU = core.use_gpu()
    io.logger_setup()

    sessions = ['000541', '000472', '000692', '000715']
    all_files, all_images_2d, all_images_3d, all_labels = load_data(sessions)

    for i in range(5):
        output_folder_path = f"/scratch/th3129/wormID/results/cellpose/fine_tune/fold_{i}/"
        split_file_path = "/scratch/th3129/wormID/datasets/dataset_split.csv"
        train_images, train_labels, train_files, test_images_3d, test_labels, test_files = cross_validate(all_files, all_images_2d, all_images_3d, all_labels, i, split_file_path)
        print(len(train_images), len(train_labels), len(test_images_3d), len(test_labels))
        # fine tune the model on 2d images
        model = models.CellposeModel(gpu=True,model_type=None,diam_mean=15)
        
        model_path = train.train_seg(model.net, train_data=train_images, train_labels=train_labels,
                                    channels=[[0, 0]], normalize=True, 
                                    test_data=None, test_labels=None,
                                    weight_decay=1e-4, SGD=True, learning_rate=0.07,
                                    n_epochs=50, model_name="cellpose_retrained")
        
        # evaluate the model by running inference on 2d slices, then stitch together to get 3d predictions
        trained_model = models.CellposeModel(pretrained_model=model_path, gpu=True)
        masks, flows, styles = trained_model.eval(test_images_3d, diameter=15, cellprob_threshold=-5, 
                                                min_size=10,channels=[[0, 0]], normalize=True, do_3D=True)
        visualize_flows(flows[0])

        for i in range(len(test_images_3d)):
            file = os.path.basename(test_files[i])
            saved_path = os.path.join(output_folder_path, os.path.splitext(file)[0]+'.png')
            np.save(os.path.splitext(saved_path)[0]+'.npy', masks[i])
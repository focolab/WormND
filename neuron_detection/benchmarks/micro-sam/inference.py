from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from pathlib import Path

from micro_sam import util
from micro_sam.util import get_sam_model, get_device, precompute_image_embeddings
from micro_sam.multi_dimensional_segmentation import mask_data_to_segmentation, _preprocess_closing
from micro_sam.instance_segmentation import AMGBase
from micro_sam import instance_segmentation

from typing import Optional, Union, Tuple
import elf.tracking.tracking_utils as track_utils
import elf.segmentation as seg_utils
from skimage.measure import regionprops

import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import nifty
import torch

def automatic_3d_segmentation(
    volume: np.ndarray,
    predictor: SamPredictor,
    segmentor: AMGBase,
    embedding_path: Optional[Union[str, os.PathLike]] = None,
    with_background: bool = True,
    gap_closing: Optional[int] = None,
    min_z_extent: Optional[int] = None,
    verbose: bool = True,
    **kwargs,
) -> np.ndarray:
    """Segment volume in 3d.

    First segments slices individually in 2d and then merges them across 3d
    based on overlap of objects between slices.

    Args:
        volume: The input volume.
        predictor: The SAM model.
        segmentor: The instance segmentation class.
        embedding_path: The path to save pre-computed embeddings.
        with_background: Whether the segmentation has background.
        gap_closing: If given, gaps in the segmentation are closed with a binary closing
            operation. The value is used to determine the number of iterations for the closing.
        min_z_extent: Require a minimal extent in z for the segmented objects.
            This can help to prevent segmentation artifacts.
        verbose: Verbosity flag.
        kwargs: Keyword arguments for the 'generate' method of the 'segmentor'.

    Returns:
        The segmentation.
    """
    offset = 0
    seg_shape = volume[:, :, :, 0].shape if len(volume.shape) == 4 else volume.shape
    segmentation = np.zeros(seg_shape, dtype="uint32")

    min_object_size = kwargs.pop("min_object_size", 0)
    image_embeddings = precompute_image_embeddings(predictor, volume, save_path=embedding_path, ndim=3)

    for i in tqdm(range(segmentation.shape[0]), desc="Segment slices", disable=not verbose):
        segmentor.initialize(volume[i], image_embeddings=image_embeddings, verbose=False, i=i)
        seg = segmentor.generate(**kwargs)
        if len(seg) == 0:
            continue
        else:
            seg = mask_data_to_segmentation(seg, with_background=with_background, min_object_size=min_object_size)
            max_z = seg.max()
            if max_z == 0:
                continue
            seg[seg != 0] += offset
            offset = max_z + offset
        segmentation[i] = seg

    segmentation = merge_instance_segmentation_3d(
        segmentation, beta=0.5, with_background=with_background, gap_closing=gap_closing, min_z_extent=min_z_extent
    )

    return segmentation


def merge_instance_segmentation_3d(
    slice_segmentation: np.ndarray,
    beta: float = 0.5,
    with_background: bool = True,
    gap_closing: Optional[int] = None,
    min_z_extent: Optional[int] = None,
    verbose: bool = True,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
) -> np.ndarray:
    """Merge stacked 2d instance segmentations into a consistent 3d segmentation.

    Solves a multicut problem based on the overlap of objects to merge across z.

    Args:
        slice_segmentation: The stacked segmentation across the slices.
            We assume that the segmentation is labeled consecutive across z.
        beta: The bias term for the multicut. Higher values lead to a larger
            degree of over-segmentation and vice versa.
        with_background: Whether this is a segmentation problem with background.
            In that case all edges connecting to the background are set to be repulsive.
        gap_closing: If given, gaps in the segmentation are closed with a binary closing
            operation. The value is used to determine the number of iterations for the closing.
        min_z_extent: Require a minimal extent in z for the segmented objects.
            This can help to prevent segmentation artifacts.
        verbose: Verbosity flag.
        pbar_init: Callback to initialize an external progress bar. Must accept number of steps and description.
            Can be used together with pbar_update to handle napari progress bar in other thread.
            To enables using this function within a threadworker.
        pbar_update: Callback to update an external progress bar.

    Returns:
        The merged segmentation.
    """
    _, pbar_init, pbar_update, pbar_close = util.handle_pbar(verbose, pbar_init, pbar_update)

    if gap_closing is not None and gap_closing > 0:
        pbar_init(slice_segmentation.shape[0] + 1, "Merge segmentation")
        slice_segmentation = _preprocess_closing(slice_segmentation, gap_closing, pbar_update)
    else:
        pbar_init(1, "Merge segmentation")

    # Extract the overlap between slices.
    edges = track_utils.compute_edges_from_overlap(slice_segmentation, verbose=False)

    uv_ids = np.array([[edge["source"], edge["target"]] for edge in edges])
    overlaps = np.array([edge["score"] for edge in edges])

    n_nodes = int(slice_segmentation.max() + 1)
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)

    costs = seg_utils.multicut.compute_edge_costs(overlaps)
    # set background weights to be maximally repulsive
    if with_background:
        bg_edges = (uv_ids == 0).any(axis=1)
        costs[bg_edges] = -8.0

    node_labels = seg_utils.multicut.multicut_decomposition(graph, 1.0 - costs, beta=beta)

    segmentation = nifty.tools.take(node_labels, slice_segmentation)

    if min_z_extent is not None and min_z_extent > 0:
        props = regionprops(segmentation)
        filter_ids = []
        for prop in props:
            box = prop.bbox
            z_extent = box[3] - box[0]
            if z_extent < min_z_extent:
                filter_ids.append(prop.label)
        if filter_ids:
            segmentation[np.isin(segmentation, filter_ids)] = 0

    pbar_update(1)
    pbar_close()

    return segmentation

if __name__ == "__main__":
    sessions = ['000541','000472','000692','000715']
    
    for session in sessions:
        train_input_folder_path = f'/scratch/th3129/wormID/datasets/tiff_files/{session}/train'
        test_input_folder_path = f'/scratch/th3129/wormID/datasets/tiff_files/{session}/test'
        output_folder_path = f"/scratch/th3129/wormID/results/micro-sam/{session}"

        train_files = [os.path.join(train_input_folder_path, file) for file in os.listdir(train_input_folder_path) if file.endswith("_img.tiff")]
        test_files = [os.path.join(test_input_folder_path, file) for file in os.listdir(test_input_folder_path) if file.endswith("_img.tiff")]
        files = train_files + test_files
        
        imgs = [tiff.imread(f) for f in files]
        imgs = [np.transpose(img, (0, 2, 3, 1)) for img in imgs]

        predictor, decoder = instance_segmentation.get_predictor_and_decoder(model_type="vit_b_lm", checkpoint_path=None)
        segmentor = instance_segmentation.InstanceSegmentationWithDecoder(predictor, decoder)

        for i in range(len(imgs)):
            instances = automatic_3d_segmentation(
                volume=imgs[i], 
                predictor=predictor, 
                segmentor=segmentor
            )
            
            file = os.path.basename(files[i])
            saved_path = os.path.join(output_folder_path, os.path.splitext(file)[0]+'.npy')
            np.save(saved_path, instances)
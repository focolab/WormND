# coding: utf-8

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import dask.array as da
import scipy.ndimage as ndi
from napari.utils.notebook_display import nbscreenshot
from rich.pretty import pprint
from tifffile import imread

from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatcher
from traccuracy.matchers._base import Matched, Matcher
from traccuracy.metrics import CTCMetrics, DivisionMetrics
from traccuracy._tracking_graph import TrackingGraph
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from ultrack import MainConfig, add_flow, segment, link, solve, track, to_tracks_layer, tracks_to_zarr, to_ctc 
from ultrack.utils import estimate_parameters_from_labels, labels_to_contours
from ultrack.utils.array import array_apply, create_zarr
from ultrack.imgproc import robust_invert, detect_foreground, normalize
from ultrack.imgproc.flow import timelapse_flow, advenct_from_quasi_random, trajectories_to_tracks
from ultrack.utils.cuda import on_gpu
from ultrack.tracks.stats import tracks_df_movement

from stardist.models import StarDist3D
from stardist.nms import _ind_prob_thresh

import os
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from glob import glob
import functools

class StarDist3DCustom(StarDist3D):

    def _predict_instances_generator(self, img, axes=None, normalizer=None,
                                     sparse=True,
                                     prob_thresh=None, nms_thresh=None,
                                     scale=None,
                                     n_tiles=None, show_tile_progress=True,
                                     verbose=False,
                                     return_labels=True,
                                     predict_kwargs=None, nms_kwargs=None,
                                     overlap_label=None, return_predict=False):
        """Predict instance segmentation from input image.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Input image
        axes : str or None
            Axes of the input ``img``.
            ``None`` denotes that axes of img are the same as denoted in the config.
        normalizer : :class:`csbdeep.data.Normalizer` or None
            (Optional) normalization of input image before prediction.
            Note that the default (``None``) assumes ``img`` to be already normalized.
        sparse: bool
            If true, aggregate probabilities/distances sparsely during tiled
            prediction to save memory (recommended).
        prob_thresh : float or None
            Consider only object candidates from pixels with predicted object probability
            above this threshold (also see `optimize_thresholds`).
        nms_thresh : float or None
            Perform non-maximum suppression that considers two objects to be the same
            when their area/surface overlap exceeds this threshold (also see `optimize_thresholds`).
        scale: None or float or iterable
            Scale the input image internally by this factor and rescale the output accordingly.
            All spatial axes (X,Y,Z) will be scaled if a scalar value is provided.
            Alternatively, multiple scale values (compatible with input `axes`) can be used
            for more fine-grained control (scale values for non-spatial axes must be 1).
        n_tiles : iterable or None
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that are processed independently and re-assembled.
            This parameter denotes a tuple of the number of tiles for every image axis (see ``axes``).
            ``None`` denotes that no tiling should be used.
        show_tile_progress: bool
            Whether to show progress during tiled prediction.
        verbose: bool
            Whether to print some info messages.
        return_labels: bool
            Whether to create a label image, otherwise return None in its place.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of Keras model.
        nms_kwargs: dict
            Keyword arguments for non-maximum suppression.
        overlap_label: scalar or None
            if not None, label the regions where polygons overlap with that value
        return_predict: bool
            Also return the outputs of :func:`predict` (in a separate tuple)
            If True, implies sparse = False

        Returns
        -------
        (:class:`numpy.ndarray`, dict), (optional: return tuple of :func:`predict`)
            Returns a tuple of the label instances image and also
            a dictionary with the details (coordinates, etc.) of all remaining polygons/polyhedra.

        """
        if predict_kwargs is None:
            predict_kwargs = {}
        if nms_kwargs is None:
            nms_kwargs = {}

        if return_predict and sparse:
            sparse = False
            warnings.warn("Setting sparse to False because return_predict is True")

        nms_kwargs.setdefault("verbose", verbose)

        _axes         = self._normalize_axes(img, axes)
        _axes_net     = self.config.axes
        _permute_axes = self._make_permute_axes(_axes, _axes_net)
        _shape_inst   = tuple(s for s,a in zip(_permute_axes(img).shape, _axes_net) if a != 'C')

        if scale is not None:
            if isinstance(scale, numbers.Number):
                scale = tuple(scale if a in 'XYZ' else 1 for a in _axes)
            scale = tuple(scale)
            len(scale) == len(_axes) or _raise(ValueError(f"scale {scale} must be of length {len(_axes)}, i.e. one value for each of the axes {_axes}"))
            for s,a in zip(scale,_axes):
                s > 0 or _raise(ValueError("scale values must be greater than 0"))
                (s in (1,None) or a in 'XYZ') or warnings.warn(f"replacing scale value {s} for non-spatial axis {a} with 1")
            scale = tuple(s if a in 'XYZ' else 1 for s,a in zip(scale,_axes))
            verbose and print(f"scaling image by factors {scale} for axes {_axes}")
            img = ndi.zoom(img, scale, order=1)

        yield 'predict'  # indicate that prediction is starting
        res = None
        if sparse:
            for res in self._predict_sparse_generator(img, axes=axes, normalizer=normalizer, n_tiles=n_tiles,
                                                      prob_thresh=prob_thresh, show_tile_progress=show_tile_progress, **predict_kwargs):
                if res is None:
                    yield 'tile'  # yield 'tile' each time a tile has been processed
        else:
            for res in self._predict_generator(img, axes=axes, normalizer=normalizer, n_tiles=n_tiles,
                                               show_tile_progress=show_tile_progress, **predict_kwargs):
                if res is None:
                    yield 'tile'  # yield 'tile' each time a tile has been processed
            res = tuple(res) + (None,)

        if self._is_multiclass():
            prob, dist, prob_class, points, prob_map = res
        else:
            prob, dist, points, prob_map = res
            prob_class = None

        yield 'nms'  # indicate that non-maximum suppression is starting
        res_instances = self._instances_from_prediction(_shape_inst, prob, dist,
                                                        points=points,
                                                        prob_class=prob_class,
                                                        prob_thresh=prob_thresh,
                                                        nms_thresh=nms_thresh,
                                                        scale=(None if scale is None else dict(zip(_axes,scale))),
                                                        return_labels=return_labels,
                                                        overlap_label=overlap_label,
                                                        **nms_kwargs)

        # last "yield" is the actual output that would have been "return"ed if this was a regular function
        if return_predict:
            yield res_instances, tuple(res[:-1]), prob_map
        else:
            yield res_instances, prob_map

    @functools.wraps(_predict_instances_generator)
    def predict_instances(self, *args, **kwargs):
        # the reason why the actual computation happens as a generator function
        # (in '_predict_instances_generator') is that the generator is called
        # from the stardist napari plugin, which has its benefits regarding
        # control flow and progress display. however, typical use cases should
        # almost always use this function ('predict_instances'), and shouldn't
        # even notice (thanks to @functools.wraps) that it wraps the generator
        # function. note that similar reasoning applies to 'predict' and
        # 'predict_sparse'.

        # return last "yield"ed value of generator
        r = None
        for r in self._predict_instances_generator(*args, **kwargs):
            pass
        return r

    def _predict_sparse_generator(self, img, prob_thresh=None, axes=None, normalizer=None, n_tiles=None, show_tile_progress=True, b=2, **predict_kwargs):
        """ Sparse version of model.predict()
        Returns
        -------
        (prob, dist, [prob_class], points)   flat list of probs, dists, (optional prob_class) and points
        """
        if prob_thresh is None: prob_thresh = self.thresholds.prob

        x, axes, axes_net, axes_net_div_by, _permute_axes, resizer, n_tiles, grid, grid_dict, channel, predict_direct, tiling_setup =             self._predict_setup(img, axes, normalizer, n_tiles, show_tile_progress, predict_kwargs)

        def _prep(prob, dist):
            prob = np.take(prob,0,axis=channel)
            dist = np.moveaxis(dist,channel,-1)
            dist = np.maximum(1e-3, dist)
            return prob, dist

        proba, dista, pointsa, prob_class = [],[],[], []

        if np.prod(n_tiles) > 1:
            raise NotImplementedError("The prediction function has not been realized when np.prod(n_tiles)>1")
            tile_generator, output_shape, create_empty_output = tiling_setup()

            sh = list(output_shape)
            sh[channel] = 1;

            proba, dista, pointsa, prob_classa = [], [], [], []

            for tile, s_src, s_dst in tile_generator:

                results_tile = predict_direct(tile)

                # account for grid
                s_src = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_src,axes_net)]
                s_dst = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_dst,axes_net)]
                s_src[channel] = slice(None)
                s_dst[channel] = slice(None)
                s_src, s_dst = tuple(s_src), tuple(s_dst)

                prob_tile, dist_tile = results_tile[:2]
                prob_tile, dist_tile = _prep(prob_tile[s_src], dist_tile[s_src])

                bs = list((b if s.start==0 else -1, b if s.stop==_sh else -1) for s,_sh in zip(s_dst, sh))
                bs.pop(channel)
                inds   = _ind_prob_thresh(prob_tile, prob_thresh, b=bs)
                proba.extend(prob_tile[inds].copy())
                dista.extend(dist_tile[inds].copy())
                _points = np.stack(np.where(inds), axis=1)
                offset = list(s.start for i,s in enumerate(s_dst))
                offset.pop(channel)
                _points = _points + np.array(offset).reshape((1,len(offset)))
                _points = _points * np.array(self.config.grid).reshape((1,len(self.config.grid)))
                pointsa.extend(_points)

                if self._is_multiclass():
                    p = results_tile[2][s_src].copy()
                    p = np.moveaxis(p,channel,-1)
                    prob_classa.extend(p[inds])
                yield  # yield None after each processed tile

        else:
            # predict_direct -> prob, dist, [prob_class if multi_class]
            results = predict_direct(x)
            prob, dist = results[:2]
            prob, dist = _prep(prob, dist)
            inds   = _ind_prob_thresh(prob, prob_thresh, b=b)
            proba = prob[inds].copy()
            dista = dist[inds].copy()
            _points = np.stack(np.where(inds), axis=1)
            pointsa = (_points * np.array(self.config.grid).reshape((1,len(self.config.grid))))

            if self._is_multiclass():
                p = np.moveaxis(results[2],channel,-1)
                prob_classa = p[inds].copy()

        proba = np.asarray(proba)
        dista = np.asarray(dista).reshape((-1,self.config.n_rays))
        pointsa = np.asarray(pointsa).reshape((-1,self.config.n_dim))

        prob_map = resizer.after(prob[:, :, :, None], self.config.axes)[..., 0]

        idx = resizer.filter_points(x.ndim, pointsa, axes_net)
        proba = proba[idx]
        dista = dista[idx]
        pointsa = pointsa[idx]

        # last "yield" is the actual output that would have been "return"ed if this was a regular function
        if self._is_multiclass():
            prob_classa = np.asarray(prob_classa).reshape((-1,self.config.n_classes+1))
            prob_classa = prob_classa[idx]
            yield proba, dista, prob_classa, pointsa, prob_map
        else:
            prob_classa = None
            yield proba, dista, pointsa, prob_map
            
            
def stardist_predict(frame: np.ndarray, model: StarDist3D) -> np.ndarray:
    """Normalizes and computes stardist prediction."""
    frame = normalize(frame, gamma=1.0)
    (labels, details), prob_map = model.predict_instances(frame, prob_thresh=0.1, 
                                                          verbose=False, show_tile_progress=False,)
    return labels     


class CTC_Centroid_Matcher(Matcher):
    """Match graph nodes based on measure used in cell tracking challenge benchmarking.

    A computed marker (segmentation) is matched to a reference marker if the computed
    marker covers a majority of the reference marker.

    Each reference marker can therefore only be matched to one computed marker, but
    multiple reference markers can be assigned to a single computed marker.

    See https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144959
    for complete details.
    """

    def __init__(self, distance_threshold, scale=(1.0, 1.0, 1.0)):
        # distance_threshold (float): Maximum distance between centroids to be considered a match
        # scale (tuple): scaling factor for the x, y, z dimensions (anisotropy)

        # note that distance_threshold is in real isotropic units
        self.distance_threshold = distance_threshold
        self.scale = scale

    def _compute_frame_mapping(
        self, frame: int, gt_graph, pred_graph
    ):
        """Compute the distance matrix between centroids of gt and pred nodes"""
        gt_nodes = np.array(list(gt_graph.nodes_by_frame[frame]))
        pred_nodes = np.array(list(pred_graph.nodes_by_frame[frame]))

        gt_locs = (
            np.array([gt_graph.get_location(node) for node in gt_nodes]) * self.scale
        )
        pred_locs = (
            np.array([pred_graph.get_location(node) for node in pred_nodes])
            * self.scale
        )
        distance_matrix = cdist(gt_locs, pred_locs)

        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        distances = distance_matrix[row_ind, col_ind]
        is_valid = distances <= self.distance_threshold

        gt_nodes = gt_nodes[row_ind[is_valid]]
        pred_nodes = pred_nodes[col_ind[is_valid]]

        return gt_nodes, pred_nodes, distances[is_valid]

    def _compute_mapping(self, gt_graph, pred_graph):
        """Run ctc matching

        Args:
            gt_graph (TrackingGraph): Tracking graph object for the gt
            pred_graph (TrackingGraph): Tracking graph object for the pred

        Returns:
            traccuracy.matchers.Matched: Matched data object containing the CTC mapping
        """
        mapping = []
        # Get overlaps for each frame
        for t in tqdm(
            range(gt_graph.start_frame, gt_graph.end_frame),
            desc="Matching frames",
        ):
            gt_match, pred_match, _ = self._compute_frame_mapping(
                t, gt_graph, pred_graph
            )
            for gt_node, pred_node in zip(gt_match, pred_match):
                mapping.append((gt_node, pred_node))

        return Matched(gt_graph, pred_graph, mapping)
    
    
def generate_pred_track(reference_frame_gt, pred_tracks_df, num_frame=None, xyz_resol=(1, 1, 1)):
    xyz_resol = np.array(xyz_resol)
    if num_frame is None:
        num_frame = pred_tracks_df.t.max() + 1
    
    track_id2neuron_id = {}
    pred_array = np.tile(reference_frame_gt, (num_frame, 1, 1))
    for index, row in pred_tracks_df.iterrows():
        k = int(row['t'])
        new_coor = np.array([row['x'], row['y'], row['z']])
        track_id = row['track_id']
        
        # Filter the DataFrame to get the rows with t == k
        df_k = pred_tracks_df[pred_tracks_df['t'] == k]
        
        # Calculate the Euclidean distance between new_coor and each row in df_k
        scaled_diff = (pred_array[k, :] - new_coor) * xyz_resol
        distances = np.linalg.norm(scaled_diff, axis=1)
        
        # Find the index of the nearest row
        if track_id in track_id2neuron_id:
            nearest_idx = track_id2neuron_id[track_id]
            pred_array[k:, nearest_idx, :] = new_coor
        elif np.min(distances) < 2:
            nearest_idx = np.argmin(distances)
            track_id2neuron_id[track_id] = nearest_idx
            pred_array[k:, nearest_idx, :] = new_coor
    
    return tracks_to_tracking_graph(pred_array), pred_array


def tracks_to_tracking_graph(tracks):
    """
    Convert calcium segmentation data into a traccuracy TrackingGraph.

    Parameters:
    tracks (numpy array): Array with shape (1500, 118, 3) containing
                          the time (first dimension), objects (second dimension),
                          and XYZ coordinates (third dimension).

    Returns:
    traccuracy.TrackingGraph: A tracking graph built from the provided calcium segmentation data.
    """
    graph = nx.DiGraph()
    num_time_points, num_objects, _ = tracks.shape

    # Create unique identifiers for each object at each time point
    object_ids = {(t, obj): f"object_{t}_{obj}" for t in range(num_time_points) for obj in range(num_objects)}

    # Add nodes to the graph
    for t in range(num_time_points):
        for obj in range(num_objects):
            x, y, z = tracks[t, obj]
            graph.add_node(object_ids[(t, obj)], x=x, y=y, z=z, t=t)

    # Add edges to the graph
    for t in range(num_time_points - 1):
        for obj in range(num_objects):
            current_id = object_ids[(t, obj)]
            next_id = object_ids[(t + 1, obj)]
            if current_id in graph.nodes and next_id in graph.nodes:
                graph.add_edge(current_id, next_id)

    tracking_graph = TrackingGraph(graph, location_keys=("x", "y", "z"))

    return tracking_graph

def evaluate_tracks(tracking_graph, gt_tracking_graph, distance_threshold, scale):
    metrics = run_metrics(
        gt_tracking_graph,
        tracking_graph,
        matcher=CTC_Centroid_Matcher(distance_threshold, scale),
        metrics=[CTCMetrics()],
    )
    return metrics



def main(data_directory):
    dataset_id2xyres = {
        "EY": 0.27,
        "KK": 0.32,
        "SK1": 0.3208,
        "SK2": 0.3208,
    }
    dataset_id2zres = {
        "EY": 1.5,
        "KK": 1.5,
        "SK1": 2.5,
        "SK2": 1.5,
    }

    dataset_id_list = sorted(dataset_id2xyres.keys())
    print(dataset_id_list)

    print(f'{data_directory}/*/*.h5')
    h5_filenames = glob(f'{data_directory}/*/*.h5')
    print(h5_filenames)

    summary_df = pd.read_csv('dataset_split.csv')
    summary_df['basename'] = summary_df['filename'].apply(lambda x:x.split('/')[-1].split('.')[0])

    path2fold = {}
    for path in tqdm(h5_filenames):
        basename = path.split('/')[-1].split('.h5')[0]
        idx = summary_df.loc[summary_df.basename==basename].index
        path2fold[path] = int(summary_df.loc[idx, 'dataset_split'].item())

    path2tra3, path2det3, path2tra6, path2det6 = {}, {}, {}, {}

    for fold_idx in range(5):
        fold_h5_filenames = [fn for fn in h5_filenames if path2fold[fn]==fold_idx]
        
        model = StarDist3DCustom(None, name="stardist", 
                                 basedir=f"./stardist_weights/retrained_fold{fold_idx}")
        
        for path in fold_h5_filenames:
            with h5py.File(path, 'r') as file:
                # Read the data
                calcium_image = file['calcium_image'][:]
                calcium_image = np.transpose(calcium_image, (0, 3, 2, 1)) # ultrack takes (t, z, y, x)
                calcium_segmentation = file['calcium_segmentation'][:] # t, xyz
                
                dataset_id = path.split('/')[-2]
                zyx_resol = (dataset_id2zres[dataset_id], dataset_id2xyres[dataset_id], dataset_id2xyres[dataset_id], )
                xyz_resol = (dataset_id2xyres[dataset_id], dataset_id2xyres[dataset_id], dataset_id2zres[dataset_id], )

                stardist_labels = np.zeros_like(calcium_image, dtype=np.int16)
                array_apply(
                    calcium_image,
                    out_array=stardist_labels,
                    func=stardist_predict,
                    model=model,
                )
                detection, boundaries = labels_to_contours(stardist_labels, sigma=0.0)

                cfg = MainConfig()
                cfg.data_config.n_workers = 8
                cfg.segmentation_config.n_workers = 8
                cfg.segmentation_config.min_area = 250
                cfg.segmentation_config.max_area = 15_000
                cfg.linking_config.n_workers = 12
                cfg.linking_config.max_neighbors = 5
                cfg.linking_config.max_distance = 10.0
                cfg.linking_config.distance_weight = 0.0001
                cfg.tracking_config.window_size = 70
                cfg.tracking_config.overlap_size = 5
                cfg.tracking_config.appear_weight = -0.01
                cfg.tracking_config.disappear_weight = -0.001
                cfg.tracking_config.division_weight = 0

                track(cfg, detection=detection, edges=boundaries, overwrite=True)

                tracks_df, graph = to_tracks_layer(cfg)

                gt_track = tracks_to_tracking_graph(calcium_segmentation)
                pred_track, pred_array = generate_pred_track(calcium_segmentation[0], tracks_df, xyz_resol=xyz_resol)
                
                # Save pred_track to a pickle file
                pred_filename = f"./pred_track/{path.split('/')[-2]}_{path.split('/')[-1].replace('.h5', '.npy')}"
                np.save(pred_filename, pred_array)
                    
                metrics3 = evaluate_tracks(pred_track, gt_track, 3, xyz_resol)[0]
                metrics6 = evaluate_tracks(pred_track, gt_track, 6, xyz_resol)[0]

                path2tra3[path] = metrics3['results']['TRA']
                path2det3[path] = metrics3['results']['DET']
                path2tra6[path] = metrics6['results']['TRA']
                path2det6[path] = metrics6['results']['DET']

    print(path2tra3, path2det3, path2tra6, path2det6)

    for path in tqdm(h5_filenames):
        basename = path.split('/')[-1].split('.h5')[0]
        idx = summary_df.loc[summary_df.basename==basename].index

        summary_df.loc[idx, ['tra3', 'det3','tra6','det6',] ] = [path2tra3[path], path2det3[path], 
                                                                path2tra6[path], path2det6[path] ]
    summary_df.to_csv('ultrack_metrics_finetuned.csv', index=False)

    fold_metric_df = summary_df.groupby('dataset_split').agg(
        {
            'tra3': lambda x: np.nanmean(x),
            'det3': lambda x: np.nanmean(x),
            'tra6': lambda x: np.nanmean(x),
            'det6': lambda x: np.nanmean(x)
        }
    ).reset_index()
    print(fold_metric_df)

    fold_metric_values = fold_metric_df[['tra3', 'det3', 'tra6', 'det6']]

    # Calculate nanmean and nanstd of the mean values
    nanmean_metric_values = fold_metric_values.apply(np.nanmean)
    nanstd_metric_values = fold_metric_values.apply(np.nanstd)

    # Combine the results into a new DataFrame
    result_df = pd.DataFrame({
        'metric': ['tra3', 'det3', 'tra6', 'det6'],
        'nanmean': nanmean_metric_values.values,
        'nanstd': nanstd_metric_values.values
    })
    print(result_df)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ultrack with retrained stardist weights.")
    parser.add_argument('data_directory', type=str, help='Path to the data directory')
    args = parser.parse_args()
    
    main(args.data_directory)





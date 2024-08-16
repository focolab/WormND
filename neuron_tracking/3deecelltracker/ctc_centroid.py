# modified from
# https://github.com/Janelia-Trackathon-2023/traccuracy/blob/46b99883e335eee0b0ded2c7175e5bdc7762d81f/src/traccuracy/matchers/_ctc.py
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import warnings

if TYPE_CHECKING:
    from traccuracy._tracking_graph import TrackingGraph

from traccuracy.matchers._base import Matched, Matcher


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
        # scale (tuple): scaling factor for the z, y, x dimensions (anisotropy)

        # note that distance_threshold is in real isotropic units
        self.distance_threshold = distance_threshold
        self.scale = scale

    def _compute_frame_mapping(
        self, frame: int, gt_graph: TrackingGraph, pred_graph: TrackingGraph
    ):
        """Compute the distance matrix between centroids of gt and pred nodes"""
        try:
            gt_nodes = np.array(list(gt_graph.nodes_by_frame[frame]))
            pred_nodes = np.array(list(pred_graph.nodes_by_frame[frame]))
        except KeyError:
            warnings.warn(f"No nodes found for frame {frame}")
            return [], [], []

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

    def _compute_mapping(self, gt_graph: TrackingGraph, pred_graph: TrackingGraph):
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

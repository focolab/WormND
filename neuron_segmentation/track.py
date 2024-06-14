import numpy as np
import btrack
import btrack.datasets

import traccuracy
import networkx as nx

from ctc_centroid import CTC_Centroid_Matcher
from traccuracy.metrics import CTCMetrics, DivisionMetrics

# NOTE: due to incompatible glibc versions, need to compile from source
# https://github.com/quantumjot/btrack/issues/198


def seg_to_objects(seg):
    id = 0
    objects = []
    for t, seg_t in enumerate(seg):
        for seg_t_i in seg_t:
            objects.append(
                btrack.btypes.PyTrackObject.from_dict(
                    {
                        "ID": id,
                        "x": seg_t_i[0],
                        "y": seg_t_i[1],
                        "z": seg_t_i[2],
                        "t": t,
                    }
                )
            )
            id += 1
    return objects


def bayesian_track(shape, seg, verbose=True):
    # NOTE: shape is [time, x, y, z]
    # seg is expected to be a list of [num_detect, [x,y,z + ...]]

    min_xyz = np.min(np.concatenate(seg, axis=0), axis=0)
    max_xyz = np.max(np.concatenate(seg, axis=0), axis=0)

    objects = seg_to_objects(seg)

    # initialise a tracker session using a context manager
    with btrack.BayesianTracker(verbose=verbose) as tracker:
        # NOTE: cell splitting is not considered (only tracking nuclei)
        # configure the tracker using a config file
        # disable P_branch and P_merge
        # remove P_dead, tracklet should not terminate without leaving frame
        # increase max_lost to 100
        tracker.configure("cell_config.json")

        # append the objects to be tracked
        tracker.append(objects)

        # set the volume
        assert 0 <= min_xyz[0] <= max_xyz[0] < shape[1]
        assert 0 <= min_xyz[1] <= max_xyz[1] < shape[2]
        assert 0 <= min_xyz[2] <= max_xyz[2] < shape[3]
        tracker.volume = ((0, shape[1]), (0, shape[2]), (0, shape[3]))

        # track them (in interactive mode)
        tracker.track(step_size=100)

        # generate hypotheses and run the global optimizer
        tracker.optimize()

        tracks = tracker.tracks

        # store the data in an HDF5 file
        data, properties, graph = btrack.utils.tracks_to_napari(tracks)
        np.savez("tracks.npz", data=data, properties=properties, graph=graph)

    return tracks


def tracks_to_tracking_graph(tracks):
    # load btrack Tracklets into a traccuracy TrackingGraph
    # NOTE: only objects in the tracks are considered (some might be ignored by the tracker)
    # this also might add any dummy nodes not originally detected
    graph = nx.DiGraph()

    for track in tracks:
        for i in range(len(track)):
            graph.add_node(
                track.refs[i], x=track.x[i], y=track.y[i], z=track.z[i], t=track.t[i]
            )
        for i in range(len(track) - 1):
            edge = (track.refs[i], track.refs[i + 1])
            assert edge[0] in graph.nodes and edge[1] in graph.nodes
            graph.add_edge(*edge)

    tracking_graph = traccuracy.TrackingGraph(graph, location_keys=("x", "y", "z"))

    return tracking_graph


def labels_to_tracking_graph(seg, labels):
    objects = seg_to_objects(seg)

    # NOTE: assumes that the nth object is the same in all time frames
    assert (
        len(set([tuple(x) for x in labels])) == 1
    ), "Labels are not consistent across time frames"
    assert len(objects) == sum(
        [len(x) for x in labels]
    ), "Number of objects and labels do not match"

    num_neurons = len(labels[0])

    graph = nx.DiGraph()
    for object in objects:
        graph.add_node(object.ID, x=object.x, y=object.y, z=object.z, t=object.t)

    for i in range(num_neurons):
        for j in range(i, len(objects) - num_neurons, num_neurons):
            edge = (objects[j].ID, objects[j + num_neurons].ID)
            assert edge[0] in graph.nodes and edge[1] in graph.nodes
            graph.add_edge(*edge)

    tracking_graph = traccuracy.TrackingGraph(graph, location_keys=("x", "y", "z"))

    return tracking_graph


def evaluate_tracks(tracking_graph, gt_tracking_graph, distance_threshold, scale):
    metrics = traccuracy.run_metrics(
        gt_tracking_graph,
        tracking_graph,
        matcher=CTC_Centroid_Matcher(distance_threshold, scale),
        metrics=[CTCMetrics()],
    )
    return metrics

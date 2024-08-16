import os
import glob
import warnings
import numpy as np
from dataloader import SampleData
from tqdm import tqdm
import pandas as pd
import argparse
import time
import random

from collections import defaultdict

import imageio


def round_up_to_odd(f):
    # https://stackoverflow.com/questions/31648729/round-a-float-up-to-next-odd-integer
    return np.ceil(f) // 2 * 2 + 1


def get_segmentation_kernel(radius, anisotropy):
    # radius in real units
    # anisotropy in real units
    shape = radius / anisotropy * 2
    shape = [round_up_to_odd(s) for s in shape]

    zz, yy, xx = np.mgrid[: shape[0], : shape[1], : shape[2]]
    dist = (
        ((zz - shape[0] // 2) / (shape[0] // 2)) ** 2
        + ((yy - shape[1] // 2) / (shape[1] // 2)) ** 2
        + ((xx - shape[2] // 2) / (shape[2] // 2)) ** 2
    )
    ellipse = dist <= 1

    return ellipse, dist


def calcium_seg_to_volume_seg(calcium_seg, shape, radius, anisotropy):
    # [time, x, y, z]
    assert len(shape) == 4
    assert len(calcium_seg) == shape[0]
    kernel, kernel_dist = get_segmentation_kernel(radius, anisotropy)
    assert all([k % 2 == 1 for k in kernel.shape])

    print("Converting calcium_seg to volume_seg...")
    max_label = max([len(calcium_seg[t]) for t in range(len(calcium_seg))]) + 1
    assert max_label < 256
    # populate it with inf
    dist = np.full(shape, np.inf, dtype=float)
    vol = np.zeros(shape, dtype=np.uint8)
    for t in tqdm(range(len(calcium_seg)), leave=False):
        for idx in range(calcium_seg[t].shape[0]):
            x, y, z = calcium_seg[t][idx][:3].astype(int)
            assert 0 <= x < shape[1]
            assert 0 <= y < shape[2]
            assert 0 <= z < shape[3]
            # NOTE: remember to +1
            # now stamp kernel onto vol, centered at z, y, x
            x_start = max(x - kernel.shape[0] // 2, 0)
            x_end = min(x + kernel.shape[0] // 2 + 1, shape[1])
            y_start = max(y - kernel.shape[1] // 2, 0)
            y_end = min(y + kernel.shape[1] // 2 + 1, shape[2])
            z_start = max(z - kernel.shape[2] // 2, 0)
            z_end = min(z + kernel.shape[2] // 2 + 1, shape[3])

            vol_slice = np.s_[t, x_start:x_end, y_start:y_end, z_start:z_end]
            kernel_slice = np.s_[
                kernel.shape[0] // 2
                - (x - x_start) : kernel.shape[0] // 2
                + (x_end - x),
                kernel.shape[1] // 2
                - (y - y_start) : kernel.shape[1] // 2
                + (y_end - y),
                kernel.shape[2] // 2
                - (z - z_start) : kernel.shape[2] // 2
                + (z_end - z),
            ]
            # check if overlap
            overlap = np.sum((vol[vol_slice] * kernel[kernel_slice]) > 0)
            # NOTE: muting overlapping warnings
            # if overlap > 0:
            #     warnings.warn(f"Overlap at {t}, {x}, {y}, {z}, {overlap} voxels")
            overwrite = dist[vol_slice] > kernel_dist[kernel_slice]
            vol[vol_slice] = (
                vol[vol_slice] * ~overwrite
                + kernel[kernel_slice] * (idx + 1) * overwrite
            ).astype(np.uint8)
            dist[vol_slice] = np.minimum(dist[vol_slice], kernel_dist[kernel_slice])
    return vol


class FakeProofedCoordsVol:
    def __init__(self):
        self.interpolation_factor = 1
        self.voxel_size = np.array([1, 1, 1])


class Inference3DeeCellTracker:
    # NOTE: our data is in [t, x, y, z]
    # 3Dcelltracker is in [z, y, x] # where first dimension is smaller than second dimension
    # NOTE: this is why weird transposing is done during inference/loading
    # transposed so that smaller dimension is first
    def __init__(
        self,
        sample_id,
        sample_data,
        output_dir,
        prob_thresh=None,
        digits=5,
        radius=2.0,
    ):
        self.sample_id = sample_id
        self.sample_data = sample_data
        self.output_dir = output_dir

        self.prob_thresh = prob_thresh

        self.digits = digits
        self.radius = radius

    def do_transpose(self):
        assert self.sample_data.has_calcium
        return self.sample_data.calcium.shape[1] > self.sample_data.calcium.shape[2]

    def export_inference(self):
        assert self.sample_data.has_calcium
        assert len(str(max(self.sample_data.calcium.shape))) <= self.digits

        inference_dir = os.path.join(self.output_dir, self.sample_id)
        os.makedirs(inference_dir, exist_ok=True)

        calcium = self.sample_data.calcium
        # time, x, y, z
        assert len(calcium.shape) == 4, "Calcium shape is not 4D"
        for t in tqdm(range(calcium.shape[0]), leave=False):
            for z in range(calcium.shape[3]):
                image = calcium[t, :, :, z]
                assert np.min(image) >= 0
                # rescale to 0-255
                if np.max(image) > 0:
                    image = image / np.max(image) * 255
                image = image.astype(np.uint8)
                # NOTE: transpose done here, since shape is consistent over timesteps, don't need to worry
                if image.shape[0] > image.shape[1]:
                    image = np.transpose(image, (1, 0))
                assert image.shape[0] <= image.shape[1]

                path = os.path.join(
                    inference_dir,
                    f"{self.sample_id}_t{t:0{self.digits}}_z{z:0{self.digits}}.tif",
                )
                imageio.imwrite(path, image)

    def do_inference(self, prefix, stardist_model_name, basedir):
        assert prefix in ["pretrained", "finetuned"]
        import CellTracker.stardistwrapper as sdw

        model = sdw.load_stardist_model(model_name=stardist_model_name, basedir=basedir)
        if self.prob_thresh is not None:
            # override the pretrained model's thresholding probability
            nms = model.thresholds.nms
            new_thresh = type(model.thresholds)(prob=self.prob_thresh, nms=nms)
            model._thresholds = new_thresh
            assert model.thresholds.prob == self.prob_thresh
            print(f"Using threshold: {self.prob_thresh}")
        else:
            print(f"Using default threshold: {model.thresholds.prob}")

        images_path = os.path.join(
            self.output_dir, self.sample_id, f"*t%0{self.digits}d*.tif"
        )
        # predict_and_save already attaches the "seg" folder
        path_results = os.path.join(self.output_dir, self.sample_id, prefix)
        sdw.predict_and_save(
            images_path=images_path, model=model, results_folder=path_results
        )

    def do_tracking(
        self, basedir, prefix, key="seg", recompute=False, max_detections=200
    ):
        # do matching, and export as traccuracy TrackingGraph
        import CellTracker.trackerlite as trl
        import networkx as nx
        import traccuracy

        path_results = os.path.join(self.output_dir, self.sample_id, prefix)
        ffn_model_name = "ffn_worm3"
        fake_proofed_coords_vol = FakeProofedCoordsVol()
        tracker = trl.TrackerLite(
            results_dir=path_results,
            ffn_model_name=ffn_model_name,
            proofed_coords_vol1=fake_proofed_coords_vol,
            basedir=basedir,
        )

        coords_files = sorted(glob.glob(os.path.join(path_results, key, "coords*.npy")))
        if not recompute:
            if os.path.exists(os.path.join(path_results, key, "tracking_graph.npz")):
                old_coords_files = np.load(
                    os.path.join(path_results, key, "tracking_graph.npz"),
                    allow_pickle=True,
                )["coords_files"].tolist()
                if old_coords_files == coords_files:
                    print("Skipping tracking, already computed")
                    return np.load(
                        os.path.join(path_results, key, "tracking_graph.npz"),
                        allow_pickle=True,
                    )["tracking_graph"].item()

        # assert consecutive timesteps
        for i in range(len(coords_files)):
            assert f"{i:0{self.digits}}" in coords_files[i]

        assert len(coords_files) > 1

        graph = nx.DiGraph()
        ids = {}
        for i in tqdm(range(len(coords_files) - 1), leave=False):
            t1 = i
            t2 = i + 1
            # pairs are in the form (t1, t2)
            t1_points, t2_points, pairs = match_by_ffn(
                tracker, t1, t2, max_detections=max_detections
            )
            t1_points = t1_points._raw
            t2_points = t2_points._raw

            # undo transpose, back to original coordinates
            if self.do_transpose():
                # [n, [x, y, z]] -> [n, [y, x, z]]
                t1_points = t1_points[:, [1, 0, 2]]
                t2_points = t2_points[:, [1, 0, 2]]

            for j in range(len(t1_points)):
                x, y, z = t1_points[j]
                t = t1
                if (x, y, z, t) not in ids:
                    new_id = len(ids)
                    graph.add_node(new_id, x=x, y=y, z=z, t=t)
                    ids[(x, y, z, t)] = new_id
            for j in range(len(t2_points)):
                x, y, z = t2_points[j]
                t = t2
                if (x, y, z, t) not in ids:
                    new_id = len(ids)
                    graph.add_node(new_id, x=x, y=y, z=z, t=t)
                    ids[(x, y, z, t)] = new_id
            for j in range(len(pairs)):
                x, y, z = t1_points[pairs[j][0]]
                t = t1
                id1 = ids[(x, y, z, t)]
                x, y, z = t2_points[pairs[j][1]]
                t = t2
                id2 = ids[(x, y, z, t)]
                assert id1 in graph.nodes and id2 in graph.nodes
                graph.add_edge(id1, id2)
        tracking_graph = traccuracy.TrackingGraph(graph, location_keys=("x", "y", "z"))

        np.savez(
            os.path.join(path_results, key, "tracking_graph.npz"),
            tracking_graph=tracking_graph,
            coords_files=coords_files,
        )

        return tracking_graph

    def export_training(self):
        assert self.sample_data.has_calcium
        assert len(str(max(self.sample_data.calcium.shape))) <= self.digits

        training_dir = os.path.join(self.output_dir, "all")
        raw_path = os.path.join(training_dir, "raw")
        label_path = os.path.join(training_dir, "label")
        os.makedirs(raw_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)

        calcium = self.sample_data.calcium
        seg_map = calcium_seg_to_volume_seg(
            self.sample_data.calcium_seg,
            self.sample_data.calcium.shape,
            self.radius,
            anisotropy=self.sample_data.scale,
        )

        assert len(calcium.shape) == 4, "Calcium shape is not 4D"
        # NOTE: 3deecelltracker expects dimensions in [z, y, x]
        for t in tqdm(range(calcium.shape[0]), leave=False):
            # [x, y, z]
            image = calcium[t]
            label = seg_map[t]
            assert image.shape == label.shape
            assert np.min(image) >= 0
            if np.max(image) > 0:
                image = image / np.max(image) * 255
                image = image.astype(np.uint8)
            # NOTE: transpose done here, since shape is consistent over timesteps, don't need to worry
            if image.shape[0] > image.shape[1]:
                image = np.transpose(image, (2, 1, 0))
                label = np.transpose(label, (2, 1, 0))
            else:
                image = np.transpose(image, (2, 0, 1))
                label = np.transpose(label, (2, 0, 1))
            assert image.shape[0] <= image.shape[1] <= image.shape[2]
            assert image.shape == label.shape

            name = os.path.join(
                f"{self.sample_id}_t{t:0{self.digits}}.tif",
            )
            imageio.volwrite(os.path.join(raw_path, name), image)
            imageio.volwrite(os.path.join(label_path, name), label)


def match_by_ffn(tracker, t1: int, t2: int, max_detections, confirmed_coord_t1=None):
    # modify match_by_ffn to allow segmented_pos_t1 and segmented_pos_t2 to have less than K_POINTS + 1 points
    import CellTracker.trackerlite as trl
    from CellTracker.trackerlite import simple_match, K_POINTS
    from CellTracker.ffn import initial_matching_ffn, normalize_points

    assert t2 not in tracker.miss_frame
    segmented_pos_t1 = tracker._get_segmented_pos(t1)
    segmented_pos_t2 = tracker._get_segmented_pos(t2)

    segmented_pos_t1._raw = segmented_pos_t1._raw[:max_detections]
    segmented_pos_t2._raw = segmented_pos_t2._raw[:max_detections]

    t1_original_len = segmented_pos_t1._raw.shape[0]
    t2_original_len = segmented_pos_t2._raw.shape[0]

    if t1_original_len == 0 or t2_original_len == 0:
        warnings.warn("No detections found")
        return segmented_pos_t1, segmented_pos_t2, []

    # duplicate it until it has K_POINTS + 1 points
    # add random noise to prevent degeneracy
    if t1_original_len < K_POINTS + 1:
        segmented_pos_t1._raw = np.concatenate(
            [segmented_pos_t1._raw] * int(np.ceil((K_POINTS + 1) / t1_original_len)),
            axis=0,
        )
        segmented_pos_t1._raw[t1_original_len:] += np.random.normal(
            scale=1, size=segmented_pos_t1._raw[t1_original_len:].shape
        )
    if t2_original_len < K_POINTS + 1:
        segmented_pos_t2._raw = np.concatenate(
            [segmented_pos_t2._raw] * int(np.ceil((K_POINTS + 1) / t2_original_len)),
            axis=0,
        )
        segmented_pos_t2._raw[t2_original_len:] += np.random.normal(
            scale=1, size=segmented_pos_t2._raw[t2_original_len:].shape
        )

    if confirmed_coord_t1 is None:
        confirmed_coord_t1 = segmented_pos_t1

    confirmed_coords_norm_t1, (mean_t1, scale_t1) = normalize_points(
        confirmed_coord_t1.real, return_para=True
    )
    segmented_coords_norm_t2 = (segmented_pos_t2.real - mean_t1) / scale_t1

    matching_matrix = initial_matching_ffn(
        tracker.ffn_model, confirmed_coords_norm_t1, segmented_coords_norm_t2, K_POINTS
    )

    # NOTE: matching_matrix has shape (t2, t1)
    # now compress the matching matrix
    assert matching_matrix.shape[0] == segmented_pos_t2._raw.shape[0]
    assert matching_matrix.shape[1] == segmented_pos_t1._raw.shape[0]
    assert matching_matrix.shape[0] % t2_original_len == 0
    assert matching_matrix.shape[1] % t1_original_len == 0
    while matching_matrix.shape[0] > t2_original_len:
        matching_matrix[:t2_original_len, :] += matching_matrix[-t2_original_len:, :]
        matching_matrix = matching_matrix[:-t2_original_len, :]
    while matching_matrix.shape[1] > t1_original_len:
        matching_matrix[:, :t1_original_len] += matching_matrix[:, -t1_original_len:]
        matching_matrix = matching_matrix[:, :-t1_original_len]
    segmented_pos_t1._raw = segmented_pos_t1._raw[:t1_original_len]
    segmented_pos_t2._raw = segmented_pos_t2._raw[:t2_original_len]

    try:
        _, pairs_px2 = simple_match(matching_matrix)
    except Exception as e:
        print(e)
        warnings.warn("ignoring failed match")
        pairs_px2 = []

    return segmented_pos_t1, segmented_pos_t2, pairs_px2


def get_sample_id(file):
    path_parts = file.split(os.sep)
    dataset = path_parts[-3]
    fileid = path_parts[-2]

    sample_id = f"{dataset}-{fileid}"
    assert "_" not in sample_id

    return sample_id


def export_inference(inference_path, files, sample_ids):
    # assert uniqueness
    assert len(set(sample_ids)) == len(
        sample_ids
    ), "Sample IDs are not unique, rewrite get_sample_id"
    for file, sample_id in tqdm(zip(files, sample_ids), leave=False):
        if os.path.exists(os.path.join(inference_path, sample_id)):
            warnings.warn(f"Skipping {sample_id}, assuming already exported")
            continue
        sample_data = SampleData(file, load_pal=False, load_calcium=True)
        assert sample_data.has_calcium
        inference = Inference3DeeCellTracker(sample_id, sample_data, inference_path)
        inference.export_inference()


def inference(
    inference_path, prob_thresh, use_pretrained, dataset_split, training_path
):
    sample_ids = sorted(os.listdir(inference_path))
    dataset_sample_ids = [get_sample_id(file) for file in dataset_split["filename"]]
    for sample_id in tqdm(sample_ids, leave=False):
        sample_data = None  # inference methods don't use sample_data
        inference = Inference3DeeCellTracker(
            sample_id, sample_data, inference_path, prob_thresh
        )
        if use_pretrained:
            stardist_model_name = "stardist_worm1"
            basedir = "/data/adhinart/celegans/3d_checkpoints"
        else:
            idx = dataset_sample_ids.index(sample_id)
            split = int(dataset_split.dataset_split[idx])
            stardist_model_name = f"stardist_fold{split}"
            basedir = os.path.join(training_path, "checkpoints", stardist_model_name)

        print(f"Using {stardist_model_name}")
        inference.do_inference(
            ("pretrained" if use_pretrained else "finetuned"),
            stardist_model_name,
            basedir,
        )


def parallel_inference(
    inference_path, prob_thresh, use_pretrained, dataset_split, training_path
):
    sample_ids = sorted(os.listdir(inference_path))
    dataset_sample_ids = [get_sample_id(file) for file in dataset_split["filename"]]

    # assumes that no jobs fail
    while True:
        print("waiting...")
        time.sleep(random.randint(0, 5))
        print("starting...")
        prefix = "pretrained" if use_pretrained else "finetuned"
        sample_ids = sorted(os.listdir(inference_path))
        sample_ids = [
            sample_id
            for sample_id in sample_ids
            if not os.path.exists(
                os.path.join(inference_path, sample_id, prefix, "seg")
            )
        ]
        if len(sample_ids) == 0:
            break
        print(f"{len(sample_ids)} samples left")
        sample_id = sample_ids[0]
        with open("log.txt", "a") as f:
            f.write(sample_id + "\n")
        print(f"Processing {sample_id}")
        os.makedirs(
            os.path.join(inference_path, sample_id, prefix, "seg"), exist_ok=True
        )
        sample_data = None  # inference methods don't use sample_data
        inference = Inference3DeeCellTracker(
            sample_id, sample_data, inference_path, prob_thresh
        )
        if use_pretrained:
            stardist_model_name = "stardist_worm1"
            basedir = "/data/adhinart/celegans/3d_checkpoints"
        else:
            idx = dataset_sample_ids.index(sample_id)
            split = int(dataset_split.dataset_split[idx])
            stardist_model_name = f"stardist_fold{split}"
            basedir = os.path.join(training_path, "checkpoints", stardist_model_name)
        print(f"Using {stardist_model_name}")
        inference.do_inference(
            ("pretrained" if use_pretrained else "finetuned"),
            stardist_model_name,
            basedir,
        )
        with open("log.txt", "a") as f:
            f.write(f"Finished {sample_id}\n")


def tracking(inference_path, files, sample_ids, use_pretrained):
    prefix = "pretrained" if use_pretrained else "finetuned"
    basedir = "/data/adhinart/celegans/3d_checkpoints"
    sample_ids = sorted(os.listdir(inference_path))
    for sample_id in tqdm(sample_ids, leave=False):
        file = files[sample_ids.index(sample_id)]
        sample_data = SampleData(file, load_pal=False, load_calcium=False)
        inference = Inference3DeeCellTracker(sample_id, sample_data, inference_path)
        inference.do_tracking(basedir, prefix=prefix)


def parallel_tracking(inference_path, files, sample_ids, use_pretrained, clear=False):
    prefix = "pretrained" if use_pretrained else "finetuned"
    basedir = "/data/adhinart/celegans/3d_checkpoints"
    # if clear, delete all _tracking
    if clear:
        sample_ids = sorted(os.listdir(inference_path))
        for sample_id in sample_ids:
            if os.path.exists(
                os.path.join(inference_path, sample_id, prefix, "seg", "_tracking")
            ):
                os.remove(
                    os.path.join(inference_path, sample_id, prefix, "seg", "_tracking")
                )

    # assumes that no jobs fail
    sample_ids = sorted(os.listdir(inference_path))

    for sample_id in tqdm(sample_ids, leave=False):
        print("waiting...")
        time.sleep(random.randint(0, 5))
        print("starting...")

        if os.path.exists(
            os.path.join(inference_path, sample_id, prefix, "seg", "_tracking")
        ):
            continue
        # touch tracking
        open(
            os.path.join(inference_path, sample_id, prefix, "seg", "_tracking"), "w"
        ).close()
        print(f"Processing {sample_id}")

        file = files[sample_ids.index(sample_id)]
        sample_data = SampleData(file, load_pal=False, load_calcium=False)
        inference = Inference3DeeCellTracker(sample_id, sample_data, inference_path)
        inference.do_tracking(basedir, prefix=prefix)
        # delete _tracking
        os.remove(os.path.join(inference_path, sample_id, prefix, "seg", "_tracking"))


def evaluate(inference_path, files, sample_ids, distance_thresholds, use_pretrained):
    import track

    prefix = "pretrained" if use_pretrained else "finetuned"

    tracks = sorted(
        glob.glob(
            os.path.join(inference_path, "*", prefix, "seg", "tracking_graph.npz")
        )
    )
    # ../elegans/inference/000472-sub-2022-04-26-w00-NP1/seg/tracking_graph.npz -> 000472-sub-2022-04-26-w00-NP1
    available_sample_ids = [track.split(os.sep)[-4] for track in tracks]

    if os.path.exists("new_preliminary_results.npz"):
        results = np.load("new_preliminary_results.npz", allow_pickle=True)[
            "results"
        ].item()
    else:
        results = defaultdict(dict)
    for sample_id in tqdm(available_sample_ids, leave=False):
        if sample_id in results:
            continue
        file = files[sample_ids.index(sample_id)]
        sample_data = SampleData(file, load_pal=False, load_calcium=False)

        data = np.load(
            os.path.join(
                inference_path, sample_id, prefix, "seg", "tracking_graph.npz"
            ),
            allow_pickle=True,
        )
        tracking_graph = data["tracking_graph"].item()
        coords_files = data["coords_files"].tolist()

        # NOTE: NOTE: NOTE: shorten to get what's available
        gt_tracking_graph = track.labels_to_tracking_graph(
            sample_data.calcium_seg[: len(coords_files)],
            sample_data.calcium_labels[: len(coords_files)],
        )
        for distance_threshold in distance_thresholds:
            metrics = track.evaluate_tracks(
                tracking_graph,
                gt_tracking_graph,
                distance_threshold=distance_threshold,
                scale=sample_data.scale,
            )
            print(metrics)
            results[sample_id][distance_threshold] = metrics
    np.savez(f"new_preliminary_results.npz", results=results)


def metrics(dataset_split):
    results = np.load("new_preliminary_results.npz", allow_pickle=True)[
        "results"
    ].item()
    dataset_sample_ids = [get_sample_id(file) for file in dataset_split["filename"]]

    max_sample_id = None
    max_tra_3_score = -1

    metric_splits_tra_3 = defaultdict(list)
    metric_splits_tra_6 = defaultdict(list)
    metric_splits_det_3 = defaultdict(list)
    metric_splits_det_6 = defaultdict(list)
    for sample_id in dataset_sample_ids:
        # for sample_id in results:
        idx = dataset_sample_ids.index(sample_id)
        if not (
            dataset_split["dataset_split"][idx].isdigit()  # don't use "original" split
            and dataset_split["use_for_calcium_task"][idx] == 1  # samples with calcium
            and dataset_split["dandi_id"][idx] != 776
        ):
            continue
        split = int(dataset_split.dataset_split[idx])

        if sample_id in results:
            metric_splits_tra_3[split].append(
                results[sample_id][3.0][0]["results"]["TRA"]
            )
            metric_splits_tra_6[split].append(
                results[sample_id][6.0][0]["results"]["TRA"]
            )
            metric_splits_det_3[split].append(
                results[sample_id][3.0][0]["results"]["DET"]
            )
            metric_splits_det_6[split].append(
                results[sample_id][6.0][0]["results"]["DET"]
            )
            if results[sample_id][3.0][0]["results"]["TRA"] > max_tra_3_score:
                max_tra_3_score = results[sample_id][3.0][0]["results"]["TRA"]
                max_sample_id = sample_id
        else:
            print(f"{sample_id} not in results, with split {split}")
            metric_splits_tra_3[split].append(0)
            metric_splits_tra_6[split].append(0)
            metric_splits_det_3[split].append(0)
            metric_splits_det_6[split].append(0)

    metric_splits_tra_3 = [
        np.mean(metric_splits_tra_3[split]) for split in metric_splits_tra_3
    ]
    metric_splits_tra_6 = [
        np.mean(metric_splits_tra_6[split]) for split in metric_splits_tra_6
    ]
    metric_splits_det_3 = [
        np.mean(metric_splits_det_3[split]) for split in metric_splits_det_3
    ]
    metric_splits_det_6 = [
        np.mean(metric_splits_det_6[split]) for split in metric_splits_det_6
    ]

    print(
        "metric_splits_tra_3 mean:",
        np.mean(metric_splits_tra_3),
        np.std(metric_splits_tra_3),
    )
    print(
        "metric_splits_tra_6 mean:",
        np.mean(metric_splits_tra_6),
        np.std(metric_splits_tra_6),
    )
    print(
        "metric_splits_det_3 mean:",
        np.mean(metric_splits_det_3),
        np.std(metric_splits_det_3),
    )
    print(
        "metric_splits_det_6 mean:",
        np.mean(metric_splits_det_6),
        np.std(metric_splits_det_6),
    )

    print("max_tra_3_score:", max_tra_3_score)
    print("max_sample_id:", max_sample_id)


def export_training(training_path, files, sample_ids, radius, dataset_split):
    # assert uniqueness
    assert len(set(sample_ids)) == len(
        sample_ids
    ), "Sample IDs are not unique, rewrite get_sample_id"
    for file, sample_id in tqdm(zip(files, sample_ids), leave=False):
        # this assumes no errors
        if glob.glob(os.path.join(training_path, "all", "*", f"{sample_id}_*.tif")):
            warnings.warn(f"Skipping {sample_id}, assuming already exported")
            continue
        sample_data = SampleData(file, load_pal=False, load_calcium=True)
        assert sample_data.has_calcium
        inference = Inference3DeeCellTracker(
            sample_id, sample_data, training_path, radius=radius
        )
        inference.export_training()

    # create symlinks for all of the folds
    dataset_sample_ids = [get_sample_id(file) for file in dataset_split["filename"]]

    splits = [int(x) for x in dataset_split.dataset_split if x.isdigit()]
    splits = sorted(set(splits))
    for split in splits:
        split_dir = os.path.join(training_path, str(split))
        os.makedirs(split_dir, exist_ok=True)
        os.makedirs(os.path.join(split_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "label"), exist_ok=True)

    for dir in ["raw", "label"]:
        for file in tqdm(
            sorted(glob.glob(os.path.join(training_path, "all", dir, "*.tif")))
        ):
            sample_id = os.path.basename(file).split("_")[0]
            dataset_idx = dataset_sample_ids.index(sample_id)
            split = int(dataset_split.dataset_split[dataset_idx])

            # put it in all the splits except split
            for s in splits:
                if s != split:
                    split_dir = os.path.join(training_path, str(s))
                    os.symlink(
                        file, os.path.join(split_dir, dir, os.path.basename(file))
                    )


def train(training_path, fold, epochs):
    import CellTracker.stardistwrapper as sdw
    import modified_sdw as msdw

    path_train_images = os.path.join(training_path, str(fold), "raw", "*.tif")
    path_train_labels = os.path.join(training_path, str(fold), "label", "*.tif")

    X, Y, X_trn, Y_trn, X_val, Y_val, n_channel = msdw.load_training_images(
        path_train_images, path_train_labels
    )

    model_name = f"stardist_fold{fold}"

    basedir = os.path.join(training_path, "checkpoints", model_name)
    os.makedirs(basedir, exist_ok=True)
    model = msdw.configure(Y, n_channel, model_name=model_name, basedir=basedir)

    model.train(
        X_trn,
        Y_trn,
        validation_data=(X_val, Y_val),
        augmenter=sdw.augmenter,
        epochs=epochs,
    )


def optimize_thresholds(training_path, fold):
    # NOTE: takes too long, OOM issues, skipping
    import CellTracker.stardistwrapper as sdw
    import modified_sdw as msdw

    path_train_images = os.path.join(training_path, str(fold), "raw", "*.tif")
    path_train_labels = os.path.join(training_path, str(fold), "label", "*.tif")

    X, Y, X_trn, Y_trn, X_val, Y_val, n_channel = msdw.load_training_images(
        path_train_images, path_train_labels
    )
    model_name = f"stardist_fold{fold}"
    basedir = os.path.join(training_path, "checkpoints", model_name)
    model = sdw.load_stardist_model(model_name=model_name, basedir=basedir)
    model.optimize_thresholds(tqdm(X_val), Y_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path", type=str, default="/data/projects/weilab/dataset/wormid"
    )
    parser.add_argument(
        "--inference_path", type=str, default="/data/adhinart/celegans/inference"
    )
    parser.add_argument(
        "--training_path", type=str, default="/data/adhinart/celegans/training"
    )
    parser.add_argument("--dataset_split", type=str, default="./dataset_split.csv")
    parser.add_argument("--export_inference", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--parallel_inference", action="store_true")
    parser.add_argument("--use_pretrained", action="store_true")
    # parser.add_argument("--prob_thresh", type=float, default=None)
    # parser.add_argument("--prob_thresh", type=float, default=0.01)
    parser.add_argument("--prob_thresh", type=float, default=0.4)
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--parallel_tracking", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    # parser.add_argument("--prob_thresh", type=float, default=0.637042)
    parser.add_argument("--export_training", action="store_true")
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--optimize_thresholds", action="store_true")
    # default None
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1000)

    args = parser.parse_args()
    base_path = args.base_path
    inference_path = args.inference_path
    training_path = args.training_path
    dataset_split = pd.read_csv(args.dataset_split)
    os.makedirs(inference_path, exist_ok=True)
    os.makedirs(training_path, exist_ok=True)

    files = []
    for i in range(len(dataset_split)):
        # i.e., not original split,
        if (
            dataset_split["dataset_split"][i].isdigit()  # don't use "original" split
            and dataset_split["use_for_calcium_task"][i] == 1  # samples with calcium
            and dataset_split["dandi_id"][i] != 776
        ):
            files.append(os.path.join(base_path, dataset_split["filename"][i]))

    files = sorted(files)
    sample_ids = [get_sample_id(file) for file in files]

    if args.export_inference:
        export_inference(inference_path, files, sample_ids)

    if (
        args.inference
        or args.parallel_inference
        or args.tracking
        or args.parallel_tracking
    ):
        import tensorflow as tf

        devices = tf.config.list_physical_devices("GPU")
        if len(devices) == 0:
            print("module load cuda11.2; module load cudnn8.1-cuda11.2")
            raise RuntimeError("No GPU detected, exiting...")

    if args.inference:
        inference(
            inference_path,
            args.prob_thresh,
            args.use_pretrained,
            dataset_split,
            training_path,
        )

    if args.parallel_inference:
        parallel_inference(
            inference_path,
            args.prob_thresh,
            args.use_pretrained,
            dataset_split,
            training_path,
        )

    if args.tracking:
        tracking(inference_path, files, sample_ids, args.use_pretrained)

    if args.parallel_tracking:
        parallel_tracking(
            inference_path, files, sample_ids, args.use_pretrained, clear=False
        )

    if args.evaluate:
        evaluate(
            inference_path,
            files,
            sample_ids,
            distance_thresholds=[3.0, 6.0],
            use_pretrained=args.use_pretrained,
        )

    if args.metrics:
        metrics(dataset_split)

    if args.export_training:
        export_training(training_path, files, sample_ids, args.radius, dataset_split)

    if args.train:
        fold = args.fold
        assert fold is not None
        train(training_path, fold, epochs=args.epochs)

    if args.optimize_thresholds:
        fold = args.fold
        assert fold is not None
        optimize_thresholds(training_path, fold)

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

from pynwb import NWBHDF5IO
import nrrd

import glob
import track
import warnings


class SampleData:
    def __init__(
        self,
        path,
        load_pal=True,
        load_calcium=True,
        gt_776_path="/data/projects/weilab/dataset/wormid/000776_gt",
    ):
        # load_pal/load_calcium: whether to load the image/calcium H5 datasets into memory
        self.path = path

        with NWBHDF5IO(path, mode="r", load_namespaces=True) as io:
            read_nwb = io.read()
            self.identifier = read_nwb.identifier

            # location of neuron centers why is 4 dimensional (675, 154, 21, 1.)
            self.seg = read_nwb.processing["NeuroPAL"]["NeuroPALSegmentation"][
                "NeuroPALNeurons"
            ].voxel_mask[:]
            # array full of identifiers for types of neurons
            labels = read_nwb.processing["NeuroPAL"]["NeuroPALSegmentation"][
                "NeuroPALNeurons"
            ]["ID_labels"][:]
            if isinstance(labels[0], str):
                labels = labels.tolist()
            else:
                labels = [x.tolist() for x in labels]
                labels = ["".join(x) for x in labels]
            # [num_detections], list of strings
            self.labels = labels
            # get which channels of the image correspond to which RGBW pseudocolors
            self.channels = read_nwb.acquisition["NeuroPALImageRaw"].RGBW_channels[:]
            # (Z, X, Y, channels)
            if load_pal:
                self.image = read_nwb.acquisition["NeuroPALImageRaw"].data[:]
            else:
                self.image = read_nwb.acquisition["NeuroPALImageRaw"].data

            self.scale = read_nwb.imaging_planes["NeuroPALImVol"].grid_spacing[:]

            self.has_calcium = "CalciumImageSeries" in read_nwb.acquisition
            if self.has_calcium:
                # HDF5 dataset, (time, x, y, z) (potentially multichannel)
                if load_calcium:
                    self.calcium = read_nwb.acquisition["CalciumImageSeries"].data[:]
                else:
                    self.calcium = read_nwb.acquisition["CalciumImageSeries"].data

                if len(self.calcium.shape) != 4:
                    assert len(self.calcium.shape) == 5, "Invalid calcium shape"
                    assert (
                        read_nwb.acquisition["CalciumImageSeries"]
                        .imaging_volume.optical_channel[0]
                        .name
                        == "GCaMP"
                    ), "Unknown calcium channel format"
                    warnings.warn(
                        f"Multichannel calcium imaging encountered in {path}, using only the first channel"
                    )
                    self.calcium = self.calcium[..., 0]

                if "000776" in path:
                    self.calcium_seg, self.calcium_labels = self.load_776_segmentations(
                        gt_776_path
                    )
                else:
                    assert hasattr(
                        read_nwb.processing["CalciumActivity"][
                            "CalciumSeriesSegmentation"
                        ],
                        "plane_segmentations",
                    ) ^ hasattr(
                        read_nwb.processing["CalciumActivity"][
                            "CalciumSeriesSegmentation"
                        ],
                        "data",
                    ), f"No calcium plane/volume segmentations found in {path}"

                    if hasattr(
                        read_nwb.processing["CalciumActivity"][
                            "CalciumSeriesSegmentation"
                        ],
                        "plane_segmentations",
                    ):
                        (
                            self.calcium_seg,
                            self.calcium_labels,
                        ) = self.load_plane_segmentations(path, read_nwb)
                    else:
                        warnings.warn(
                            f"Converting volume segmentations to plane segmentations in {path}"
                        )
                        (
                            self.calcium_seg,
                            self.calcium_labels,
                        ) = self.load_volume_segmentations(path, read_nwb)

    def load_plane_segmentations(self, path, read_nwb):
        # neuron centers
        calcium_seg = []
        calcium_labels = []
        for seg_tpoint in read_nwb.processing["CalciumActivity"][
            "CalciumSeriesSegmentation"
        ].plane_segmentations.keys():
            time = int(seg_tpoint.split("_")[-1])
            seg = read_nwb.processing["CalciumActivity"][
                "CalciumSeriesSegmentation"
            ].plane_segmentations[seg_tpoint]["voxel_mask"][:]
            # [num_detections, 4]
            seg = np.array([x[0].tolist() for x in seg])

            min_xyz = np.min(seg, axis=0)
            max_xyz = np.max(seg, axis=0)
            try:
                assert 0 <= min_xyz[0] <= max_xyz[0] < self.calcium.shape[1]
                assert 0 <= min_xyz[1] <= max_xyz[1] < self.calcium.shape[2]
                assert 0 <= min_xyz[2] <= max_xyz[2] < self.calcium.shape[3]
            except AssertionError:
                warnings.warn(
                    f"Calcium segmentations out of bounds in {path}, clamping coordinates"
                )
                seg[:, 0] = np.clip(seg[:, 0], 0, self.calcium.shape[1] - 1)
                seg[:, 1] = np.clip(seg[:, 1], 0, self.calcium.shape[2] - 1)
                seg[:, 2] = np.clip(seg[:, 2], 0, self.calcium.shape[3] - 1)

            # (x, y, z, weight)
            calcium_seg.append((time, seg))
            try:
                labels = read_nwb.processing["CalciumActivity"][
                    "CalciumSeriesSegmentation"
                ].plane_segmentations[seg_tpoint]["ID_labels"][:]
                labels = [x.tolist() for x in labels]
                # [num_detections], list of strings
                labels = ["".join(x) for x in labels]
            except KeyError:
                # if no labels are present
                # for example in 000541/sub-20190924-01/sub-20190924-01_ses-20190924_ophys.nwb
                warnings.warn(f"No calcium labels found in {path}, using empty labels")
                labels = [""] * seg.shape[0]
            calcium_labels.append((time, labels))
            assert seg.shape[0] == len(
                labels
            ), "Calcium segmentation and labels mismatch in number of detections"
        # to list
        calcium_seg = sorted(calcium_seg, key=lambda x: x[0])
        calcium_labels = sorted(calcium_labels, key=lambda x: x[0])
        calcium_seg = [x[1] for x in calcium_seg]
        calcium_labels = [x[1] for x in calcium_labels]

        if self.calcium.shape[0] != len(calcium_seg):
            # for example in 000541/sub-20190924-01/sub-20190924-01_ses-20190924_ophys.nwb
            warnings.warn(
                f"Calcium and segmentation mismatch in time in {path}: shape {self.calcium.shape[0]} vs num_labels {len(calcium_seg)}, using shorter of the two"
            )
            num_frames = min(self.calcium.shape[0], len(calcium_seg))
            # NOTE: this force loads calcium_seg regardless of load_calcium
            calcium_seg = calcium_seg[:num_frames]
            calcium_labels = calcium_labels[:num_frames]
            self.calcium = self.calcium[:num_frames]

        return calcium_seg, calcium_labels

    def load_volume_segmentations(self, path, read_nwb):
        # get center of mass of each neuron in each frame
        # if missing, linearly interpolate position from last/next frame
        # [time, x, y, z]
        seg = read_nwb.processing["CalciumActivity"]["CalciumSeriesSegmentation"].data[
            :
        ]
        max_id = np.max(seg)
        calcium_seg = -np.ones((seg.shape[0], max_id, 3))
        calcium_labels = [[""] * max_id for _ in range(seg.shape[0])]

        for t in range(seg.shape[0]):
            seg_t = seg[t]
            idx = np.stack(np.where(seg_t > 0), axis=-1)
            if len(idx) == 0:
                continue
            labels = seg_t[idx[:, 0], idx[:, 1], idx[:, 2]]
            order = np.argsort(labels)
            idx, labels = idx[order], labels[order]
            # https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function
            labels, index = np.unique(labels, return_index=True)
            components = np.split(idx, index[1:])
            centers = np.array([np.mean(component, axis=0) for component in components])
            for label, center in zip(labels, centers):
                assert label > 0
                calcium_seg[t, label - 1] = center
        if np.any(calcium_seg == -1):
            warnings.warn(
                f"Calcium segmentation missing neurons in {path}, {np.sum(calcium_seg == -1)/(np.prod(calcium_seg.shape) / 3)} neurons missing, interpolating tracks"
            )
            for label in range(calcium_seg.shape[1]):
                for dim in range(3):
                    calcium_seg[:, label, dim] = interpolate_missing_values(
                        calcium_seg[:, label, dim]
                    )
        calcium_seg = [calcium_seg[t].astype(int) for t in range(calcium_seg.shape[0])]

        return calcium_seg, calcium_labels

    def load_776_segmentations(self, gt_path):
        raise NotImplementedError
        # crop_params = np.load(os.path.join(gt_path, "crop_params.npy"), allow_pickle=True).item()
        # # '2022-06-14-01'
        # date = "-".join(os.path.basename(self.path).split("-")[1:5])
        # assert date in crop_params, f"Date {date} not found in 776 crop_params"
        # crop_params = crop_params[date]
        #
        # gt_path = os.path.join(gt_path, date)
        # assert os.path.exists(gt_path), f"776 GT path {gt_path} not found"
        # files = os.listdir(gt_path)
        # segs = []
        # for file in sorted(files):
        #     if file.endswith(".nrrd"):
        #         data, headers = nrrd.read(os.path.join(gt_path, file))
        #         segs.append(data)
        #
        # max_id = np.max([np.max(seg) for seg in segs])
        # calcium_seg = -np.ones((len(segs), max_id, 3))
        # calcium_labels = [[""] * max_id for _ in range(len(segs))]
        #
        # for t in range(len(segs)):
        #     seg_t = segs[t]
        #     idx = np.stack(np.where(seg_t > 0), axis=-1)
        #     if len(idx) == 0:
        #         continue
        #     labels = seg_t[idx[:, 0], idx[:, 1], idx[:, 2]]
        #     order = np.argsort(labels)
        #     idx, labels = idx[order], labels[order]
        #     # https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function
        #     labels, index = np.unique(labels, return_index=True)
        #     components = np.split(idx, index[1:])
        #     centers = np.array([np.mean(component, axis=0) for component in components])
        #     for label, center in zip(labels, centers):
        #         assert label > 0
        #         calcium_seg[t, label - 1] = center
        # __import__('pdb').set_trace()

        # if np.any(calcium_seg == -1):
        #     warnings.warn(
        #         f"Calcium segmentation missing neurons in {path}, {np.sum(calcium_seg == -1)/(np.prod(calcium_seg.shape) / 3)} neurons missing, interpolating tracks"
        #     )
        #     for label in range(calcium_seg.shape[1]):
        #         for dim in range(3):
        #             calcium_seg[:, label, dim] = interpolate_missing_values(
        #                 calcium_seg[:, label, dim]
        #             )
        # calcium_seg = [calcium_seg[t].astype(int) for t in range(calcium_seg.shape[0])]
        #
        #     return calcium_seg, calcium_labels


def interpolate_missing_values(arr):
    # converts [-1,-1,1,2, -1,3,4,-1] to [1,1,1,2,2.5,3,4,4]
    assert np.any(arr != -1), "All values are missing"
    arr = arr.copy()

    # Get the indices of the missing values
    missing_indices = np.where(arr == -1)[0]

    # Loop over each missing index to interpolate the value
    for idx in missing_indices:
        # Find the previous non-missing value
        prev_idx = idx - 1
        while prev_idx >= 0 and arr[prev_idx] == -1:
            prev_idx -= 1

        # Find the next non-missing value
        next_idx = idx + 1
        while next_idx < len(arr) and arr[next_idx] == -1:
            next_idx += 1

        # If both previous and next indices are within bounds, interpolate
        if prev_idx >= 0 and next_idx < len(arr):
            arr[idx] = arr[prev_idx] + (arr[next_idx] - arr[prev_idx]) * (
                (idx - prev_idx) / (next_idx - prev_idx)
            )
        # If only previous index is within bounds, use the previous value
        elif prev_idx >= 0:
            arr[idx] = arr[prev_idx]
        # If only next index is within bounds, use the next value
        elif next_idx < len(arr):
            arr[idx] = arr[next_idx]

    return arr


if __name__ == "__main__":
    files = sorted(glob.glob("/data/projects/weilab/dataset/wormid/*/*/*.nwb"))
    # files = [x for x in files if "000472-sub-2022-04-26-w00-NP1" in x]
    # files = ["/data/projects/weilab/dataset/wormid/000472/sub-2022-04-26-w00-NP1/sub-2022-04-26-w00-NP1_ses-20220426_ophys.nwb"]
    files = [
        "/data/projects/weilab/dataset/wormid/000692/sub-2308918-03/sub-2308918-03_ses-20230918T132100_ophys.nwb"
    ]
    # files = [x for x in files if "sub-20190929-02" in x]
    # files = [x for x in files if "000776_gt" not in x]
    # files = [files[61]]
    # files = [x for x in files if "/000776/" in x]
    # __import__('pdb').set_trace()
    # print(files[52])
    # files = [files[52]]
    # files = [
    #     "/data/projects/weilab/dataset/wormid/000541/sub-20190924-01/sub-20190924-01_ses-20190924_ophys.nwb"
    # ]

    results = {}
    for path in tqdm(files):
        data = SampleData(path, load_pal=False, load_calcium=True)
        gt_tracking_graph = track.labels_to_tracking_graph(
            data.calcium_seg, data.calcium_labels
        )
        np.savez(f"gt_tracking_graph.npz", tracking_graph=gt_tracking_graph)
        __import__("pdb").set_trace()
        np.savez(f"sample_calcium.npy", data.calcium[500])
    #     if data.has_calcium:
    #         # dummy test data to generate sample tracks from GT
    #         tracks = track.bayesian_track(
    #             data.calcium.shape, data.calcium_seg, verbose=False
    #         )
    #         tracking_graph = track.tracks_to_tracking_graph(tracks)
    #
    #         gt_tracking_graph = track.labels_to_tracking_graph(
    #             data.calcium_seg, data.calcium_labels
    #         )
    #         metrics = track.evaluate_tracks(
    #             tracking_graph,
    #             gt_tracking_graph,
    #             distance_threshold=10.0,
    #             scale=data.scale,
    #         )
    #         results["path"] = metrics
    #         print(metrics)
    # np.save("results.npy", results)

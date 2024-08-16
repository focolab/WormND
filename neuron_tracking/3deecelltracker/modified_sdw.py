import numpy as np

import os
import sys
from tqdm import tqdm
from csbdeep.utils import Path, normalize
from glob import glob
from tifffile import imread
from stardist import fill_label_holes

from tensorflow.keras.utils import Sequence


from stardist import Rays_GoldenSpiral
from stardist import fill_label_holes, calculate_extents, gputools_available
from stardist.models import Config3D
from stardist.utils import _normalize_grid

from CellTracker.stardist3dcustom import StarDist3DCustom
from joblib import Memory

STARDIST_MODELS = "stardist_models"
UP_LIMIT = 400000
CACHE_PATH = "/data/adhinart/celegans/cache"
mem = Memory(CACHE_PATH, verbose=0)

os.makedirs(CACHE_PATH, exist_ok=True)

np.random.seed(42)


def load_training_images(
    path_train_images: str, path_train_labels: str
):  # , max_projection: bool):
    # modified to not call plotting functions
    """Load images for training StarDist3DCustom"""
    # realpath so memoization properly works across splits
    X = sorted(glob(path_train_images))
    # X = sorted([os.path.realpath(p) for p in X])
    Y = sorted(glob(path_train_labels))
    # Y = sorted([os.path.realpath(p) for p in Y])
    assert len(X) > 0 and len(Y) > 0, "Error: No images found in either X or Y."
    assert all(
        Path(x).name == Path(y).name for x, y in zip(X, Y)
    ), "Error: Filenames in X and Y do not match."

    if len(X) == 1:
        print(
            "Warning: only one training data was provided! It will be used for both training and validation purposes!"
        )
        X = [X[0], X[0]]
        Y = [Y[0], Y[0]]

    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    X = DataLoader(X, "X")
    Y = DataLoader(Y, "Y")
    X_trn = DataLoader(X_trn, "X")
    Y_trn = DataLoader(Y_trn, "Y")
    X_val = DataLoader(X_val, "X")
    Y_val = DataLoader(Y_val, "Y")
    print("number of images: %3d" % len(X))
    print("- training:       %3d" % len(X_trn))
    print("- validation:     %3d" % len(X_val))
    print(f"X[0].shape={X[0].shape}")

    n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]
    axis_norm = (0, 1, 2)  # normalize channels independently
    # axis_norm = (0,1,2,3) # normalize channels jointly
    if n_channel > 1:
        print(
            "Normalizing image channels %s."
            % ("jointly" if axis_norm is None or 3 in axis_norm else "independently")
        )
        sys.stdout.flush()

    # i = 0
    # img, lbl = X[i], Y[i]
    # assert img.ndim in (3, 4)
    # img = img if img.ndim == 3 else img[..., :3]
    # if max_projection:
    #     plot_img_label_max_projection(img, lbl)
    # else:
    #     plot_img_label_center_slice(img, lbl)

    return X, Y, X_trn, Y_trn, X_val, Y_val, n_channel


class DataLoader(Sequence):
    # https://github.com/stardist/stardist/issues/57#issuecomment-880645603

    def __init__(self, paths, mode):
        assert mode in ("X", "Y")
        self.paths, self.mode = paths, mode

        # self.ndim = self.__getitem__(0).ndim
        # self.dtype = self.__getitem__(0).dtype

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.mode == "X":
            return get_x(self.paths[idx])
        if self.mode == "Y":
            return get_y(self.paths[idx])


@mem.cache
def get_x(path):
    X = imread(path)
    axis_norm = (0, 1, 2)  # normalize channels independently
    # axis_norm = (0,1,2,3) # normalize channels jointly

    X = normalize(X, 1, 99.8, axis=axis_norm)
    assert X.ndim in (3, 4)
    X = X if X.ndim == 3 else X[..., :3]
    return X


@mem.cache
def get_y(path):
    Y = imread(path)
    Y = fill_label_holes(Y)
    return Y


def print_dict(my_dict: dict):
    for key, value in my_dict.items():
        print(f"{key}: {value}")


def configure(
    Y,
    n_channel: int,
    up_limit: int = UP_LIMIT,
    model_name: str = "stardist",
    basedir: str = STARDIST_MODELS,
):
    extents = calculate_extents(Y)
    anisotropy = tuple(np.max(extents) / extents)
    print("empirical anisotropy of labeled objects = %s" % str(anisotropy))

    # 96 is a good default choice (see 1_data.ipynb)
    n_rays = 96

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = False and gputools_available()

    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)

    # Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

    # Set train_patch_size which should
    # 1. match anisotropy and under a predefined limitation
    a, b, c = anisotropy
    train_patch_size = np.cbrt(up_limit * a * b * c) / np.array([a, b, c])
    # 2. less than the image size
    up_limit_xyz = Y[0].shape[0], np.min(Y[0].shape[1:3]), np.min(Y[0].shape[1:3])
    scaling = np.min(np.asarray(up_limit_xyz) / train_patch_size)
    if scaling < 1:
        train_patch_size = train_patch_size * scaling
    # 3. can be divided by div_by (related to unet architecture)
    unet_n_depth = 2  #
    grid_norm = _normalize_grid(grid, 3)
    unet_pool = 2, 2, 2
    div_by = tuple(p**unet_n_depth * g for p, g in zip(unet_pool, grid_norm))
    print(f"div_by={div_by}")
    train_patch_size = [int(d * (i // d)) for i, d in zip(train_patch_size, div_by)]
    # 4. size of x and y should be the same (since augmentation will flip x-y axes)
    train_patch_size[1] = train_patch_size[2] = min(train_patch_size[1:])

    conf = Config3D(
        rays=rays,
        grid=grid,
        anisotropy=anisotropy,
        use_gpu=use_gpu,
        n_channel_in=n_channel,
        # adjust for your data below (make patch size as large as possible)
        train_patch_size=train_patch_size,
        train_batch_size=2,
        # https://github.com/stardist/stardist/issues/107#issuecomment-791410065
        # so memory does not grow during training
        train_sample_cache=False,
    )
    print_dict(vars(conf))
    assert conf.unet_n_depth == unet_n_depth
    assert conf.grid == grid_norm
    assert conf.unet_pool == unet_pool

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory

        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        limit_gpu_memory(0.8)
        # alternatively, try this:
        # limit_gpu_memory(None, allow_growth=True)

    model = StarDist3DCustom(conf, name=model_name, basedir=basedir)

    median_size = calculate_extents(Y, np.median)
    fov = np.array(model._axes_tile_overlap("ZYX"))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print(
            "WARNING: median object size larger than field of view of the neural network."
        )

    return model

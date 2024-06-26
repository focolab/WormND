# WormND
This repository contains the code necessary to reproduce the results and analyses in the WormND NeurIPS 2024 Benchmark paper.

To clone this repository with all included submodules, run git clone --recurse-submodules in the command line. You can also selectively clone specific submodules by running git submodule update --init submoduleName. 

Note that the package requirements should be listed in each submodule, but you may need to generate different environments for each submodule.

## Task 1: Neuron center detection 

In this task, we will take NeuroPAL image as input, and predicts the location of each neuron. We evaluated two classical models for cell segmentation tasks, cellpose and micro-SAM (from Segment Anything). To run the code, please do the following step

1. download the data from [wormid website](https://www.wormid.org/datasets) to store in neuron_detection/datasets
2. preprocess it with preprocess.py
3. run inference.py and evaluate.py on preprocessed data in the folder for each individual model
4. aggregate the results with evaluate.py in neuron_detection/datasets

## Task 2: NeuroPAL neuron identification

ID_benchmark.ipynb contains the code for running the analyses presented in the benchmark paper once results have been produced from each model.

### CPD
  We use the implementation of coherent point drift used in the fDNC paper (https://elifesciences.org/articles/66410). This code can be found in the fDNC_neuron_ID submodule included here.  

To reproduce the benchmarking results, use the run_CPD.ipynb script to generate the results csv files. Relative data paths are in the "run_CPD.ipynb" file. These paths assume that the data folder is at the root level of this repository. They may need to be changed if the data is in another place on your local file system.


### Statistical Atlas
  The code for this analysis can be found in the NWBelegans repository presented in Sprague et al. (https://www.biorxiv.org/content/10.1101/2024.04.28.591397v1), added as a submodule here. Further details on the model itself can be found in the original Statistical Atlas paper (https://link.springer.com/chapter/10.1007/978-3-030-59722-1_12).

The NWB_atlas.ipynb script contains the code for retraining the statistical atlas and generating the benchmarking results. Further details on this can be found in the NWBelegans ReadMe and within the jupyter notebook file itself.
- 

### CRF_ID
  Matlab code for CRF_ID is provided as a submodule. To learn about details of the method, please read the published paper (https://elifesciences.org/articles/60321). To learn about details of how to run the code in general, please read the readme document inside the CRF_ID folder.

To produce results in the WormND benchmark paper, the following files have been used:
- Command script:
    - “command_script_neuroPAL.m” in “Main” folder for running the functions 
- New atlases: 
    - atlas_o.mat: atlas built using the 10 original neuroPAL worms; no color correction
    - atlas_o_cc.mat: atlas built using the 10 original neuroPAL worms; color corrected
    - atlas_oXXXX: atlas built using the 10 original neuroPAL worms plus 4 of the k-fold cross-validation groups.
- New datasets:
    - “NeuroPAL_corpus” folder in “Datasets”
 
### fDNC

### Code for summary benchmarks

## Task 3: Neuron tracking in whole-brain calcium imaging

In this task, we will take calcium video data (GCaMP) and the ground truth neuron locations at the reference frame (i.e., the 1st frame) as input, and predict the location of each neuron for the subsequent frames. We evaluated two classical models for cell tracking tasks: Ultrack and 3DeeCellTracker. Detailed instructions are included in the individual folder of each tracking method.

### Ultrack

The segmentation part in Ultrack can be performed using watershed, Cellpose, or Stardist. Here, we use Stardist with the pretrained weights available at [OSF](https://osf.io/pgr95/). The pretrained and finetuned weights can also be downloaded from [Google Drive](https://drive.google.com/drive/folders/15mRWmyQd7sy58BETUSrmaSb-QD6f8GYY?usp=sharing).

To run the pretrained version of Ultrack, please execute:

```console
python ultrack-stardist-pretrained.py /path/to/calcium_h5
```


To run the finetuned version of Ultrack, please execute:

```console
python ultrack-stardist-finetuned.py /path/to/calcium_h5
```



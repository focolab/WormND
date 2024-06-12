# WormND
Code for reproducing results and figures for the WormND benchmark paper.

To clone this repository with included submodules, run git clone --recurse-submodules in the command line.

## Task 1: Neuron center detection 

## Task 2: NeuroPAL neuron identification

### CPD
  We use the implementation of coherent point drift used in the fDNC paper (https://elifesciences.org/articles/66410). This code can be found in the fDNC_neuron_ID submodule included here.  

To reproduce the benchmarking results, use the run_CPD.ipynb script in fDNC_Neuron_ID/src to generate the results csv files. Relative data paths are in the "run_CPD.ipynb" file. These paths assume that the data folder is at the root level of this repository. They may need to be changed if the data is in another place on your local file system.


### Statistical Atlas
  The code for this analysis can be found in the NWBelegans repository presented in Sprague et al. (https://www.biorxiv.org/content/10.1101/2024.04.28.591397v1), added as a submodule here.

To produce results in the WormND benchmark paper, the following files have been used:
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

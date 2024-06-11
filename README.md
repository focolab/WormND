# WormND
Code for reproducing results and figures for the WormND benchmark paper.

## Task 1: Neuron center detection 

## Task 2: NeuroPAL neuron identification

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


## Task 3: Neuron tracking in whole-brain calcium imaging

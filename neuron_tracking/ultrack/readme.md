# Ultrack for neuron tracking task

In task 3, we will take calcium video data (GCaMP) and the ground truth neuron locations at the reference frame (i.e., the 1st frame) as input, and predict the location of each neuron for the subsequent frames. We evaluated two classical models for cell tracking tasks: Ultrack and 3DeeCellTracker. Detailed instructions are included in the individual folder of each tracking method.


### prepare files

The segmentation part in Ultrack can be performed using watershed, Cellpose, or Stardist. Here, we use Stardist with the pretrained weights available at [OSF](https://osf.io/pgr95/). The pretrained and finetuned weights can also be downloaded from [Google Drive](https://drive.google.com/drive/folders/15mRWmyQd7sy58BETUSrmaSb-QD6f8GYY?usp=sharing).

Please organize the downloaded weights and scripts in the following structure:

-ultrack

  -- blahblah.py * 3
  
  -- dataset_split.csv
  
  -- stardist_weights

The calcium video data can be used in either of the following ways:

1. download (EY, KK, SK1, SK2) datasets in nwb format from [bandi](https://www.wormid.org/datasets);

2. stream specific subsets of the files from online using the DANDI API with codes from [this notebook](https://github.com/focolab/NWBelegans/blob/main/NWB_stream.ipynb) 

3. download the cleaned calcium_h5 folder from the [Google Drive](https://drive.google.com/drive/folders/15mRWmyQd7sy58BETUSrmaSb-QD6f8GYY?usp=sharing). These files are converted from NWB files by convert_nwb_to_h5.py, and have a size of approximately one-third of the NWB files.


### run inference

To run the pretrained version of Ultrack, please execute:

```console
python ultrack-stardist-pretrained.py /path/to/calcium_h5
```


To run the finetuned version of Ultrack, please execute:

```console
python ultrack-stardist-finetuned.py /path/to/calcium_h5
```




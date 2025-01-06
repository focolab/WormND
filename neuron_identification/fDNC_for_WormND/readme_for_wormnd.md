
# fDNC for WormND


## environment setup

```bash
conda create -n wormnd python=3.10
conda activate wormnd
gh repo clone XinweiYu/fDNC_Neuron_ID
cd fDNC_Neuron_ID
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install shapely 
```


the code is in such a structure:
```
-fDNC_Neuron_ID
    - src
    - model
    - wormnd
        - __init__.py
        - datasets_NWB.py
        - fDNC_train_for_wormnd.py
        - fDNC_predict_for_wormnd.py
        - 
        - readme_for_wormnd.md
        - pkl_data_for_fDNC
        - models_weight
        - results
```

## data prepretation

### step 1. download the data from the website
We use the data from wormid website:  
https://www.wormid.org/datasets

you can download the data from the dandi website using the following command:
it will take ~1.5T space in your computer, you may skip the SF dataset (1T) temporarily

```bash
pip install "dandi>=0.60.0"
mkdir PATH_DATA   # defined in wormnd/__init__.py
cd PATH_DATA
dandi download DANDI:000541/0.241009.1457 # EY
dandi download DANDI:000714/0.241009.1516 # HL
dandi download DANDI:000692/0.240402.2118 # KK
dandi download DANDI:000715/0.241009.1514 # NP
dandi download DANDI:000776/0.241009.1509 # SF
dandi download DANDI:000565/0.241009.1504 # SK1
dandi download DANDI:000472/0.241009.1502 # Sk2
```

the downloaded data is in such a structure:
```
- PATH_DATA
    - 000714
        - sub-1
            - sub-1_ophys.nwb
    - 000541
        - sub-20190928-08
            - sub-20190928-08_ses-20190928_ophys.nwb
```

use the **data_preview.ipynb** to show the detail of the data

### step 2. use **datasets_NWB.py** to convert the data  
   
you can use the **datasets_NWB.py** to convert the data to the format of the template worm and the test worm (fDNC)

### step 3. the synthetic data

```bash 
mkdir PATH_DATA_SYNTHETIC   # defined in wormnd/__init__.py
```

download the synthetic data from the following link:
[google drive]()  


and unzip the file to the PATH_DATA_SYNTHETIC (defined in wormnd/__init__.py)




## finetune model


there are 2 different ways to do the train test split:

1. using 5fold cross validation 
   - it will split the data into 5 folds, and use 4 folds for training and 1 fold for validation
   - it corresponds to the few-shot learning setting
2. split data based on labs
   - it will split the data based on the labs, and use the data from different labs for training and testing
   - the data from one lab will be used for testing, and the data from other labs will be used for training
   - it corresponds to the zero-shot learning setting



### using 5fold cross validation 

```bash
python ./wormnd/fDNC_train_for_wormnd.py --train_path PATH_DATA_SYNTHETIC --eval_path PATH_DATA_SYNTHETIC --model_path PATH_MODEL --data_mode train_0 --n_hidden 128 --n_layer 6 --batch_size 64 --valid_niter 3280  --p_rotate 0 --f_trans 0  --use_pretrain 1 --max_epoch 30  --model_idx 100
  
python ./wormnd/fDNC_train_for_wormnd.py --train_path PATH_DATA_SYNTHETIC --eval_path PATH_DATA_SYNTHETIC --model_path PATH_MODEL --data_mode train_1 --n_hidden 128 --n_layer 6 --batch_size 64 --valid_niter 3280  --p_rotate 0 --f_trans 0  --use_pretrain 0 --max_epoch 30  --model_idx 101 

python ./wormnd/fDNC_train_for_wormnd.py --train_path PATH_DATA_SYNTHETIC --eval_path PATH_DATA_SYNTHETIC --model_path PATH_MODEL --data_mode train_2 --n_hidden 128 --n_layer 6 --batch_size 64 --valid_niter 3280  --p_rotate 0 --f_trans 0  --use_pretrain 0 --max_epoch 30  --model_idx 102 
 
python ./wormnd/fDNC_train_for_wormnd.py --train_path PATH_DATA_SYNTHETIC --eval_path PATH_DATA_SYNTHETIC --model_path PATH_MODEL --data_mode train_3 --n_hidden 128 --n_layer 6 --batch_size 64 --valid_niter 3280  --p_rotate 0 --f_trans 0  --use_pretrain 0 --max_epoch 30  --model_idx 103 

python ./wormnd/fDNC_train_for_wormnd.py --train_path PATH_DATA_SYNTHETIC --eval_path PATH_DATA_SYNTHETIC --model_path PATH_MODEL --data_mode train_4 --n_hidden 128 --n_layer 6 --batch_size 64 --valid_niter 3280  --p_rotate 0 --f_trans 0  --use_pretrain 0 --max_epoch 30  --model_idx 104 
```


### split data based on labs

```bash
python ./wormnd/fDNC_train_for_wormnd.py --train_path PATH_DATA_SYNTHETIC --eval_path PATH_DATA_SYNTHETIC --model_path PATH_MODEL  --n_hidden 128 --n_layer 6 --batch_size 64 --valid_niter 3280  --p_rotate 0 --f_trans 0  --use_pretrain 0 --max_epoch 30 --data_mode train_wo_HL --model_idx 200 

python ./wormnd/fDNC_train_for_wormnd.py --train_path PATH_DATA_SYNTHETIC --eval_path PATH_DATA_SYNTHETIC --model_path PATH_MODEL  --n_hidden 128 --n_layer 6 --batch_size 64 --valid_niter 3280  --p_rotate 0 --f_trans 0  --use_pretrain 0 --max_epoch 30 --data_mode train_wo_EY --model_idx 201 

python ./wormnd/fDNC_train_for_wormnd.py --train_path PATH_DATA_SYNTHETIC --eval_path PATH_DATA_SYNTHETIC --model_path PATH_MODEL  --n_hidden 128 --n_layer 6 --batch_size 64 --valid_niter 3280  --p_rotate 0 --f_trans 0  --use_pretrain 0 --max_epoch 30 --data_mode train_wo_SK2 --model_idx 202 

python ./wormnd/fDNC_train_for_wormnd.py --train_path PATH_DATA_SYNTHETIC --eval_path PATH_DATA_SYNTHETIC --model_path PATH_MODEL  --n_hidden 128 --n_layer 6 --batch_size 64 --valid_niter 3280  --p_rotate 0 --f_trans 0  --use_pretrain 0 --max_epoch 30  --data_mode train_wo_SK1 --model_idx 203 

python ./wormnd/fDNC_train_for_wormnd.py --train_path PATH_DATA_SYNTHETIC --eval_path PATH_DATA_SYNTHETIC --model_path PATH_MODEL  --n_hidden 128 --n_layer 6 --batch_size 64 --valid_niter 3280  --p_rotate 0 --f_trans 0  --use_pretrain 0 --max_epoch 30  --data_mode train_wo_KK --model_idx 204 

python ./wormnd/fDNC_train_for_wormnd.py --train_path PATH_DATA_SYNTHETIC --eval_path PATH_DATA_SYNTHETIC --model_path PATH_MODEL  --n_hidden 128 --n_layer 6 --batch_size 64 --valid_niter 3280  --p_rotate 0 --f_trans 0  --use_pretrain 0 --max_epoch 30  --data_mode train_wo_SF --model_idx 205 

python ./wormnd/fDNC_train_for_wormnd.py --train_path PATH_DATA_SYNTHETIC --eval_path PATH_DATA_SYNTHETIC --model_path PATH_MODEL  --n_hidden 128 --n_layer 6 --batch_size 64 --valid_niter 3280  --p_rotate 0 --f_trans 0  --use_pretrain 0 --max_epoch 30  --data_mode train_wo_SK --model_idx 206
```

 




## pretrain and finetune model weight  
PATH_MODEL (defined in wormnd/__init__.py)
 
| Task | Model Path | Model Name |
|------|------------|------------|
| pretrain | pretrain | model.bin |
| train_0 | 100 | nitReg_nh128_nl6_ft0_datatrain_0_elam_0.1_model100_epoch30.bin |
| train_1 | 101 | nitReg_nh128_nl6_ft0_datatrain_1_elam_0.1_model101_epoch30.bin |
| train_2 | 102 | nitReg_nh128_nl6_ft0_datatrain_2_elam_0.1_model102_epoch30.bin |
| train_3 | 103 | nitReg_nh128_nl6_ft0_datatrain_3_elam_0.1_model103_epoch30.bin |
| train_4 | 104 | nitReg_nh128_nl6_ft0_datatrain_4_elam_0.1_model104_epoch30.bin |
 
| Task | Model Path | Model Name |
|------|------------|------------|
| train_wo_HL | 200 | nitReg_nh128_nl6_ft0_datatrain_wo_HL_elam_0.1_model200_epoch29.bin |
| train_wo_EY | 201 | nitReg_nh128_nl6_ft0_datatrain_wo_EY_elam_0.1_model201_epoch29.bin |
| train_wo_SK2 | 202 | nitReg_nh128_nl6_ft0_datatrain_wo_SK2_elam_0.1_model202_epoch29.bin |
| train_wo_SK1 | 203 | nitReg_nh128_nl6_ft0_datatrain_wo_SK1_elam_0.1_model203_epoch29.bin |
| train_wo_KK | 204 | nitReg_nh128_nl6_ft0_datatrain_wo_KK_elam_0.1_model204_epoch29.bin |
| train_wo_SF | 205 | nitReg_nh128_nl6_ft0_datatrain_wo_SF_elam_0.1_model205_epoch29.bin |
| train_wo_SK | 206 | nitReg_nh128_nl6_ft0_datatrain_wo_SF_elam_0.1_model206_epoch29.bin |
 

## test the performance of models 

use the **fDNC_predict_for_wormnd.py** to test the performance of the models

## combine the prediction results

to combine the prediction results to get the final score in our paper, you can use the following notebooks:

**wormnd/results/benchmark_by_lab_split/benchmark_by_lab_split.ipynb**

**wormnd/results/benchmark_by_lab_split/benchmark_by_lab_split.ipynb**

also, we provide the prediction results from the cell detection model (the result of Task1) in the following notebook:
**wormnd/results/benchmark_from_cell_detection/benchmark_from_cell_detection.ipynb**
 


















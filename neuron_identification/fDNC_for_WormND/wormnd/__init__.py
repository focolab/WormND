import os 
import sys 
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..')
add_path(lib_path)

 


## original wormid data, nwb format
PATH_DATA = r'D:\Data\WormID'

## synthetic data using current wormid data for fDNC training
PATH_DATA_SYNTHETIC = r'D:\Data\WormID\NWB_data_syn'

## converted data from NWB to pkl for fDNC training 
PATH_DATA_PKL = 'wormnd/pkl_data_for_fDNC'


# path for the model weight
PATH_MODEL = 'wormnd/models_weight'

# prediction result for different models
PATH_RESULT = 'wormnd/results'









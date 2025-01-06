from __init__ import *

import pickle
from wormnd.utils import * 
from wormnd.datasets_NWB import * 
from wormnd.datasets_NWB import get_all_files, NWB_data
from wormnd.datasets_NWB import  DATASETS_ALL, Dandi_IDS

try:
    from src.DNC_predict import predict_label  
except:
    print('fDNC_Neuron_ID is not installed, please install it first')
    # this file is from fDNC_Neuron_ID 
    # https://github.com/XinweiYu/fDNC_Neuron_ID
    # see readme_for_wormnd.md for more details


def predict_for_all_datasets(wo_norm=False, mode='from_cell_detection'): 

    path_model = PATH_MODEL
    path_data_pkl = PATH_DATA_PKL

    if mode == 'lab_split':
        task_all = [str(task) for task in range(200, 206) ]
        task_all = ['train_wo_HL', 'train_wo_EY', 'train_wo_SK2', 'train_wo_SK1', 'train_wo_KK', 'train_wo_SF']
        # task_all = ['train_wo_NP', 'train_wo_HL', 'train_wo_EY', 'train_wo_SK','train_wo_KK', 'train_wo_SF', ]
    elif mode == 'from_cell_detection':
        task_all = ['pretrain','train_0', 'train_1', 'train_2', 'train_3', 'train_4', ]
    else:
        task_all = ['pretrain','train_0', 'train_1', 'train_2', 'train_3', 'train_4', ]



    for task in task_all:
        if task == 'train_0': 
            model_path = os.path.join(path_model, '100',  f'nitReg_nh128_nl6_ft0_datatrain_0_elam_0.1_model100_epoch30.bin')
        elif task == 'train_1': 
            model_path = os.path.join(path_model, '101', f'nitReg_nh128_nl6_ft0_datatrain_1_elam_0.1_model101_epoch30.bin')
        elif task == 'train_2': 
            model_path = os.path.join(path_model, '102', f'nitReg_nh128_nl6_ft0_datatrain_2_elam_0.1_model102_epoch30.bin')
        elif task == 'train_3': 
            model_path = os.path.join(path_model, '103', f'nitReg_nh128_nl6_ft0_datatrain_3_elam_0.1_model103_epoch30.bin')
        elif task == 'train_4': 
            model_path = os.path.join(path_model, '104', f'nitReg_nh128_nl6_ft0_datatrain_4_elam_0.1_model104_epoch30.bin')
        elif task == 'pretrain':
            model_path = os.path.join(path_model, 'pretrain', 'model.bin')
            
        if task == 'train_wo_HL':
            model_path = os.path.join(path_model, '200', f'nitReg_nh128_nl6_ft0_datatrain_wo_HL_elam_0.1_model200_epoch29.bin')
        elif task == 'train_wo_EY':
            model_path = os.path.join(path_model, '201', f'nitReg_nh128_nl6_ft0_datatrain_wo_EY_elam_0.1_model201_epoch29.bin')
        elif task == 'train_wo_SK2':
            model_path = os.path.join(path_model, '202', f'nitReg_nh128_nl6_ft0_datatrain_wo_SK2_elam_0.1_model202_epoch29.bin')
        elif task == 'train_wo_SK1':
            model_path = os.path.join(path_model, '203', f'nitReg_nh128_nl6_ft0_datatrain_wo_SK1_elam_0.1_model203_epoch29.bin')
        elif task == 'train_wo_KK':
            model_path = os.path.join(path_model, '204', f'nitReg_nh128_nl6_ft0_datatrain_wo_KK_elam_0.1_model204_epoch29.bin')
        elif task == 'train_wo_SF':
            model_path = os.path.join(path_model, '205', f'nitReg_nh128_nl6_ft0_datatrain_wo_SF_elam_0.1_model205_epoch29.bin')
        elif task == 'train_wo_SK':
            model_path = os.path.join(path_model, '206', f'nitReg_nh128_nl6_ft0_datatrain_wo_SF_elam_0.1_model206_epoch29.bin')

        acc_all = []
        for dataset in DATASETS_ALL: 
            try:
                if wo_norm:
                    name = os.path.join(path_data_pkl, 'benchmark_by_pretrain_model', 'pos_col_labl__for_{dataset} (wo_norm).pkl')
                else:
                    name = os.path.join(path_data_pkl, 'benchmark_by_lab_split', 'pos_col_labl__for_{dataset}.pkl')

                if mode == 'lab_split':
                    name = os.path.join(path_data_pkl, 'benchmark_by_lab_split', 'pos_col_labl__for_{dataset}.pkl')

                if mode == 'from_cell_detection': 
                    name = os.path.join(path_data_pkl, 'benchmark_from_cell_detection', 'pos_col_labl__for_{dataset} with_label.pkl')

                data_all = np.load(name, allow_pickle=True)
            except:
                print(f'{name} is not found')
                # readme_for_wormnd.md 
                ## data preparation
                ### step 2. use **datasets_NWB.py** to convert the data 
                print('please run the following code to generate the data:')
                # path_from=PATH_DATA
                # path_to=PATH_DATA_PKL
                # need_normalization=False 
                # combined_data_all = convert_nwb_data_for_fDNC(path_from, path_to, need_normalization)
                # print(combined_data_all.keys())
                print('python datasets_NWB.py')
                assert False 

            for test_id in data_all.keys():
                test_color, test_pos, test_label_gt = data_all[test_id]
                test_color[np.isnan(test_color)] = 0
                test_pos[np.isnan(test_pos)] = 0
                if dataset=='SF':
                    test_label_gt=test_label_gt.tolist()
                for template_id in data_all.keys():
                    print(task, dataset, test_id, template_id)
                    temp_color, temp_pos, temp_label = data_all[template_id]
                    if dataset=='SF':
                        temp_label=temp_label.tolist()

                    temp_color[np.isnan(temp_color)] = 0
                    temp_pos[np.isnan(temp_pos)] = 0
                    try: 
                        test_label, candidate_list, cost = predict_label(temp_pos, temp_label, test_pos, temp_color, test_color,
                                                                         model_path=model_path, need_cost=True)
                        acc_part = []
                        acc_part.append(template_id)
                        acc_part.append(test_id)
                        
                        for k_top in range(1, 6):
                            acc_top_k_list = []
                            for label_gt, candidate_5 in zip(test_label_gt, candidate_list):
                                scores = np.asarray([score for label, score in candidate_5[:k_top]])
                                assert np.all(scores[:-1] >= scores[1:])
                                label_candidate = [label for label, score in candidate_5[:k_top] ]
                                if label_gt in label_candidate:
                                    acc_top_k_list.append(1)
                                else:
                                    acc_top_k_list.append(0)
                            acc_top_k = np.mean(acc_top_k_list)
                            acc_part.append(acc_top_k) 
                        # print(template_id, test_id, acc_top1, acc_top5)
                        acc_part.append(cost)
                    except:
                        acc_part = [template_id,test_id,0,0,100000]
                    acc_all.append(acc_part) 

        acc_all_df = pd.DataFrame(acc_all, 
                        columns=['template_id', 'test_id', 'acc_top1', 'acc_top2', 'acc_top3', 'acc_top4', 'acc_top5', 'cost'])
        if wo_norm:
            path_to = os.path.join(PATH_RESULT, 'benchmark_by_pretrain_model','result')
            os.makedirs(path_to, exist_ok=True)
            name = os.path.join(path_to, f'fDNC_{task}__acc_top1_and_top5_and_cost (wo_norm).csv')
        else:
            path_to = os.path.join(PATH_RESULT, 'benchmark_by_lab_split', 'result')
            os.makedirs(path_to, exist_ok=True)
            name = os.path.join(path_to, f'fDNC_{task}__acc_top1_and_top5_and_cost.csv')

        if mode == 'lab_split':
            path_to = os.path.join(PATH_RESULT, 'benchmark_by_lab_split', 'result')
            os.makedirs(path_to, exist_ok=True)
            name = os.path.join(path_to, f'fDNC_{task}__acc_top1_and_top5_and_cost lab_split.csv')
        if mode == 'from_cell_detection':
            path_to = os.path.join(PATH_RESULT, 'benchmark_from_cell_detection', 'result')
            os.makedirs(path_to, exist_ok=True)
            name = os.path.join(path_to, f'fDNC_{task}__acc_top1_and_top5_and_cost from_cell_detection.csv')

        acc_all_df.to_csv(name, index=False)

        print(acc_all_df)


    return acc_all_df


if __name__ == '__main__':
    
    wo_norm = True
    wo_norm = False 

    mode = ''
    mode = 'lab_split'     # mode = 'from_cell_detection'


    try: 
        # defined the PATH_DATA and PATH_MODEL and PATH_RESULT in the __init__.py
        path_from=PATH_DATA
        path_to=PATH_DATA_PKL
        need_normalization=False 
        
        for dataset in DATASETS_ALL:
            # dataset ='SF'
            if wo_norm: 
                name = os.path.join(PATH_DATA_PKL, 'benchmark_by_pretrain_model', 'pos_col_labl__for_{dataset} (wo_norm).pkl')
            else: 
                name = os.path.join(PATH_DATA_PKL, 'benchmark_by_lab_split', 'pos_col_labl__for_{dataset}.pkl')

            if mode == 'lab_split': 
                name = os.path.join(PATH_DATA_PKL, 'benchmark_by_lab_split','pos_col_labl__for_{dataset}.pkl')

            if mode == 'from_cell_detection': 
                name = os.path.join(PATH_DATA_PKL, 'benchmark_from_cell_detection','pos_col_labl__for_{dataset} with_label.pkl')
 
            data_all = np.load(name, allow_pickle=True)
    except:
        path_from=PATH_DATA
        path_to=PATH_DATA_PKL
        need_normalization=False 
        combined_data_all = convert_nwb_data_for_fDNC(path_from, path_to, need_normalization)
        print(combined_data_all.keys())




    acc_all_df = predict_for_all_datasets(wo_norm=wo_norm, mode=mode)





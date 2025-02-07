import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def extract_parts(path):
    parts = path.split('/')
    part1 = parts[0]
    part2 = parts[2].split('.')[0]
    return part1, part2

def cross_validated_result(results_df, split_file_path, split_by_lab=True):
    split_file = pd.read_csv(split_file_path)
    split_file[['session', 'worm']] = split_file['filename'].apply(lambda x: pd.Series(extract_parts(x)))

    dataset_split = 'split_by_lab' if split_by_lab else 'dataset_split'
    merged_df = pd.merge(results_df, split_file, on=['session', 'worm'])
    merged_df = merged_df[merged_df['use_for_id_task'] == 1]
    merged_df = merged_df[['worm', 'precision', 'recall', 'f1_score', dataset_split]]
    merged_df[dataset_split].astype(int)

    agg_df = merged_df.groupby(by=[dataset_split])[['precision', 'recall', 'f1_score']].mean().reset_index()
    return agg_df
    

def plot_model_comparisons(acc_stat_base, acc_stat_retrain, acc_CRF_base, acc_CRF_retrain):
    labels = ['dist=3', 'dist=6', 'dist=3', 'dist=6']
    models = ['Cellpose', 'Cellpose', 'micro-sam', 'micro-sam']
    data = {
        'Label': [],
        'Model': [],
        'Accuracy': []
    }

    for label, model, accuracies in zip(labels, models, [acc_stat_base, acc_stat_retrain, acc_CRF_base, acc_CRF_retrain]):
        data['Label'].extend([label] * len(accuracies))
        data['Model'].extend([model] * len(accuracies))
        data['Accuracy'].extend(accuracies)

    df = pd.DataFrame(data)

    fig, axs = plt.subplots()
    sns.violinplot(ax=axs, data=df, x='Label', y='Accuracy', hue='Model', cut=0, inner='point', density_norm='width')

    stats = df.groupby(['Label', 'Model'])['Accuracy'].mean().reset_index()
    xrange = [0.05, 0.257, 0.55, 0.757]

    for i, row in stats.iterrows():
        axs.axhline(y=row['Accuracy'], color='red', linestyle='-', linewidth=3, xmin=xrange[i], xmax=xrange[i]+0.2)
 
    axs.set_ylim((0, 1))
    axs.set_title('Model performance comparison')
    axs.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axs.set_xticklabels(labels)
    axs.spines[['right', 'top']].set_visible(False)
    axs.axhline(1.0, ls='--', c='grey')
    axs.axhline(0.75, ls='--', c='grey')
    axs.axhline(0.5, ls='--', c='grey')
    axs.axhline(0.25, ls='--', c='grey')

    plt.savefig('violin.png')

if __name__ == "__main__":
    models = ['cellpose', 'micro-sam']
    dists = [3, 6]
    results = []

    for model in models:
        for dist in dists:
            results_folder_path = f"/scratch/th3129/wormID/results/{model}"
            split_file_path = "/scratch/th3129/wormID/datasets/dataset_split.csv"
            dfs = []
            print(model, " dist = ", dist)

            for subdir in os.listdir(results_folder_path):
                subdir_path = os.path.join(results_folder_path, subdir)
                
                if os.path.isdir(subdir_path):
                    for file in os.listdir(subdir_path):
                        if file.endswith(f'dist{dist}.csv'):
                            df = pd.read_csv(os.path.join(subdir_path, file))
                            df['session'] = subdir
                            print(subdir, df['precision'].mean(), df['recall'].mean(), df['f1_score'].mean())
                            dfs.append(df)

            result = pd.concat(dfs, axis=0)
            excluded_sessions = []
            result = result[~result['session'].isin(excluded_sessions)]

            agg_df = cross_validated_result(result, split_file_path, split_by_lab=False)
            print("mean", agg_df['precision'].mean(), agg_df['recall'].mean(), agg_df['f1_score'].mean())
            print("std", agg_df['precision'].std(), agg_df['recall'].std(), agg_df['f1_score'].std())
            print()
            results.append(result)

    plot_model_comparisons(results[0]['f1_score'], results[1]['f1_score'], results[2]['f1_score'], results[3]['f1_score'])

import pickle
from pathlib import Path, PurePosixPath
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_write_results(dic, sample_method, dataset_name):
    p = Path(PurePosixPath('results').joinpath(dataset_name))
    p.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame.from_dict(dic)
    sample_method_name = sample_method.__name__
    file_name = PurePosixPath(p).joinpath(sample_method_name + '-' + dataset_name + '.csv')
    df.to_csv(file_name, index=False)


def write_results(results_list, dataset_name, chosen_samples):
    p = Path(PurePosixPath('results').joinpath(dataset_name))
    p.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.DataFrame.from_dict(res) for res in results_list])
    timestamp = time.strftime("%d_%m_%Y_%H%M%S")
    file_name = PurePosixPath(p).joinpath(dataset_name + timestamp + '.csv')
    df.to_csv(file_name, index=False)

    with open(str(PurePosixPath(p).joinpath(f'{dataset_name}_{timestamp}.pickle')), 'wb') as handle:
        pickle.dump(chosen_samples, handle)


def plot_sample_method(dataset_name, metric):
    import matplotlib.pyplot as plt
    from pathlib import Path, PurePosixPath
    import pandas as pd
    p = Path(PurePosixPath('results').joinpath(dataset_name))
    files = list(p.glob('*.csv'))
    method_list = []
    for file in files:
        sample_method_name = file.name.split('-')[0]
        method_list.append(sample_method_name)
        df = pd.read_csv(file)
        x = df['n_samples']
        y = df[str(metric)]
        plt.plot(x, y)
    plt.xlabel('samples')
    plt.legend(method_list)
    plt.ylabel(metric)
    plt.title(str(dataset_name))
    file_name = p.joinpath(dataset_name + '_' + metric + '.png')
    plt.savefig(file_name)
    plt.show()

    return


def pivot_table_result_by_method(result_df, metric):
    mean_by_k = pd.pivot_table(result_df, index=['sample_method', 'representation'], columns=['n_samples'],
                               values=metric, aggfunc={metric: [np.mean]}).reset_index()
    std_by_k = pd.pivot_table(result_df, index=['sample_method', 'representation'], columns=['n_samples'],
                              values=metric, aggfunc={metric: [np.std]}).reset_index()
    return mean_by_k, std_by_k




# def calculate_region_around_mean(mean_by_k, std_by_k):
#     mean_plus_std = mean_by_k['mean'] + std_by_k['std']
#     mean_minus_std = mean_by_k['mean'] - std_by_k['std']
#     return mean_minus_std, mean_plus_std

def calculate_region_around_mean(mean_by_k, std_by_k):
    from scipy.stats import  t
    confidence = 0.90
    n =5
    m = mean_by_k['mean']
    std_err = std_by_k['std']
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)

    mean_plus_std = m + h
    mean_minus_std = m - h
    return mean_minus_std, mean_plus_std

def plot_curve_with_region(mean_by_k, mean_minus_std, mean_plus_std, metric, model_type, dataset_name):
    from cycler import cycler
    from scipy.interpolate import make_interp_spline, BSpline
    import seaborn as sns
    mean_by_k.sort_values(by=['representation','sample_method'],inplace=True)
    representation_short_name = {"AvgBert":"AvB","SentenceBert":"SenB"}

    sample_method_short_name = {"lc_most_distance_2_means":"LC-DIV-Kmeans",
                                "least_confidence_k_means_sample":"LC-DBAL",
                                "least_confidence_mdr_sample":"LC-MDR",
                                "least_confidence_sample":"LC",
                                "mdr_sample":"MDR",
                                #"qbc_knn_density_sample":"QBC-KNN",
                                "random_sample":"Rand"}
    mean_by_k.replace({"sample_method":sample_method_short_name,"representation":representation_short_name},inplace=True)
    n = int(len(mean_by_k['representation'])/(mean_by_k['representation'].nunique()))
    new_colors = [plt.get_cmap('cool')(1. * i / n) for i in range(n)]
    new_colors.extend([plt.get_cmap('autumn')(1. * i / n) for i in range(n)])
    plt.rc('axes', prop_cycle=(cycler('color', new_colors)))
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    for index, row in mean_by_k.iterrows():
        x = row['mean'].index.values.astype(int)
        y = row['mean'].values
        xnew = np.linspace(x.min(), x.max(), 100)
        spl = make_interp_spline(x, y, k=2)
        power_smooth = spl(xnew)
        plt.plot(xnew, power_smooth,linewidth=0.5,markevery=3,marker = '.')
        #plt.fill_between(x, mean_minus_std.iloc[index].values, mean_plus_std.iloc[index].values, alpha=0.2)

    plt.legend(mean_by_k['sample_method'].values + '-' + mean_by_k['representation'].values,
               bbox_to_anchor=(1.05, 1.0), loc='upper left')

    plt.xlabel('samples')
    plt.ylabel(metric)
    plt.title(f'model {dataset_name}')

    p = Path(PurePosixPath('results'))
    p.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%d_%m_%Y_%H%M%S")
    file_name = PurePosixPath(p).joinpath(f'{dataset_name}/{model_type}_{metric}' + f'{timestamp}' + '.png')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()


def pivot_and_plot(result_df, metric, model_type, dataset_name):
    result_df = result_df[result_df.model_type == model_type]
    mean_by_k, std_by_k = pivot_table_result_by_method(result_df, metric)
    mean_minus_std, mean_plus_std = calculate_region_around_mean(mean_by_k, std_by_k)
    plot_curve_with_region(mean_by_k, mean_minus_std, mean_plus_std, metric, model_type, dataset_name)
    return mean_by_k



def calculate_ALC(result_df,dataset_name, metric):
    from sklearn.metrics import auc
    experiment_list = []
    ALC_list = []
    df = pd.DataFrame()
    mean_by_k, std_by_k = pivot_table_result_by_method(result_df, metric)
    for index, row in mean_by_k.iterrows():
        x = row['mean'].index.values.astype(int)
        y = row['mean'].values
        ALC = auc(x, y) / auc(x, np.array(len(x) * [1]))
        experiment_list.append(str(row['sample_method'].values + '-' + row['representation'].values))
        ALC_list.append(ALC)
    df['Experiment'] = experiment_list
    df['ALC'] = ALC_list
    p = Path(PurePosixPath('results').joinpath(dataset_name)).joinpath(time.strftime("%d_%m_%Y_%H%M%S") + 'ACL.csv')
    df.to_csv(p, index=False)

df = pd.read_csv('/Users/uri/nlp_active_learning/results/mr_sentence_polarity_embedded/mr_sentence_polarity_embedded24_04_2020_160848.csv')
pivot_and_plot(df, 'recall', 'SVC', 'mr_sentence_polarity_embedded')

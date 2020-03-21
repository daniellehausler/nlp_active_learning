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


def write_results(results_list, dataset_name):
    p = Path(PurePosixPath('results').joinpath(dataset_name))
    p.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.DataFrame.from_dict(res) for res in results_list])
    timestamp = time.strftime("%d_%m_%Y_%H%M%S")
    file_name = PurePosixPath(p).joinpath(dataset_name+ timestamp + '.csv')
    df.to_csv(file_name, index=False)


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




def pivot_table_result_by_method(result_df,metric):
    mean_by_k= pd.pivot_table(result_df,index=['sample_method','representation'],columns=['n_samples'],values=metric,aggfunc={metric: [np.mean]}).reset_index()
    std_by_k= pd.pivot_table(result_df,index=['sample_method','representation'],columns=['n_samples'],values=metric,aggfunc={metric: [np.std]}).reset_index()
    return mean_by_k , std_by_k


def calculate_region_around_mean(mean_by_k,std_by_k):
    mean_plus_std = mean_by_k['mean'] + std_by_k['std']
    mean_minus_std = mean_by_k['mean'] - std_by_k['std']
    return mean_minus_std,mean_plus_std

def plot_curve_with_region(mean_by_k,mean_minus_std,mean_plus_std,metric):
    for index,row in mean_by_k.iterrows():
        x = row['mean'].index.values.astype(int)
        y = row['mean'].values
        plt.plot(x, y)
        plt.fill_between(x,mean_minus_std.iloc[index].values,mean_plus_std.iloc[index].values,alpha=0.2)

    plt.legend(mean_by_k['sample_method'].values)
    plt.xlabel('samples')
    plt.ylabel(metric)
    plt.show()

def pivot_and_plot(result_df,metric):
    mean_by_k , std_by_k = pivot_table_result_by_method(result_df,'f1')
    mean_minus_std,mean_plus_std = calculate_region_around_mean(mean_by_k , std_by_k)
    plot_curve_with_region(mean_by_k,mean_minus_std,mean_plus_std,metric)
    return mean_by_k

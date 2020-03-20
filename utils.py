from pathlib import Path, PurePosixPath
import pandas as pd
import time



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

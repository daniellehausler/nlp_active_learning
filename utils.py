

def read_write_results(dic,sample_method,dataset_name):
    from pathlib import Path, PurePosixPath
    import pandas as pd


    p = Path(PurePosixPath('results').joinpath(dataset_name))
    p.mkdir(parents=True, exist_ok=True)

    df=pd.DataFrame.from_dict(dic)
    sample_method_name = sample_method.__name__
    file_name = PurePosixPath(p).joinpath(sample_method_name+'_'+dataset_name+'.csv')
    df.to_csv(file_name)
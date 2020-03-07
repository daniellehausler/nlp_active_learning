
import pathlib

import pandas as pd
import numpy as np

def read_write_results(dic,sample_method,dataset_name):
    #pathlib.Path('/results/').mkdir(parents=True, exist_ok=True)
    df=pd.DataFrame.from_dict(dic)
    sample_method_name = sample_method.__name__
    file_name = str(sample_method_name+'_'+dataset_name+'.csv')
    df.to_csv(file_name)
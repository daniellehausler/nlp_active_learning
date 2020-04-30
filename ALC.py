import pandas as pd
import numpy as np
from sklearn.metrics import auc
import pickle
from utils import pivot_table_result_by_method
from pathlib import Path, PurePosixPath

def calculate_ALC(result_df,dataset_name, metric):
    from sklearn.metrics import auc
    alc_results = []
    representation_short_name = {"AvgBert":"AvB","SentenceBert":"SenB"}

    sample_method_short_name = {"lc_most_distance_2_means":"LC-DIV-Kmeans",
                                "least_confidence_k_means_sample":"LC-DBAL",
                                "least_confidence_mdr_sample":"LC-MDR",
                                "least_confidence_sample":"LC",
                                "mdr_sample":"MDR",
                                "qbc_knn_density_sample":"QBC-KNN",
                                "qbc_sample":"QBC",
                                "random_sample":"Rand"}
    result_df.replace({"sample_method":sample_method_short_name,"representation":representation_short_name},inplace=True)
    representations = result_df.representation.unique().tolist()
    sample_methods = result_df.sample_method.unique().tolist()
    folds = result_df.k_fold.unique().tolist()
    for rep in representations:
        for method in sample_methods:
            alc_fold_list = [rep,method]
            for fold in folds:
                tmp_random=result_df[(result_df.representation==rep) &
                       (result_df.sample_method=='Rand') &
                       (result_df.k_fold==fold)][['n_samples',metric]]

                tmp=result_df[(result_df.representation==rep) &
                       (result_df.sample_method==method) &
                       (result_df.k_fold==fold)][['n_samples',metric]]
                x = tmp.n_samples.values
                x_rand = tmp_random.n_samples.values
                y = tmp[metric].values
                y_rand = tmp_random[metric].values
                rand_auc = auc(x_rand,y_rand)
                ALC = (auc(x, y)-rand_auc) / (auc(x, np.array(len(x) * [1]))-rand_auc)
                alc_fold_list.append(ALC)
            alc_results.append(alc_fold_list)

    df =pd.DataFrame(alc_results,columns=['Representation','Sample Method','fold1','fold2','fold3','fold4','fold5'])
    df['mean'] = df.iloc[:,2:].mean(axis=1)
    df['std'] = df.iloc[:, 2:].std(axis=1)
    df.to_csv('/Users/uri/nlp_active_learning/results/Final Results/Embedding_representation/ALC/'+dataset_name+'_'+metric+'_ALC.csv')





#df = pd.read_csv('/Users/uri/nlp_active_learning/results/toxic_5000/toxic_500026_04_2020_105244.csv')
#calculate_ALC(df,'toxic','f1')
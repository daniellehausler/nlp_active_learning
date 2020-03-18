from itertools import product
from typing import Callable, Dict, List
from utils import read_write_results, plot_sample_method, write_results
from copy import deepcopy
import pandas as pd
from active_learning import ActiveLearner
from sample_methods import *
from sklearn.model_selection import train_test_split, KFold
from model import Model
from data_pre_process import get_lstm_parsed_sentences_and_embeddings


# TODO :
#  2. add more sample methods,
#  4. parallel run of multiple experiments

def run_experiment(
        active_learner: ActiveLearner,
        model: Model,
        embeddings_train: np.array,
        train_sent: np.array,
        test_sent: np.array,
        train_y: np.array,
        test_y: np.array,
        n_iter: int,
        sample_method: Callable,
        dataset_name: str
) -> Dict:
    raw_x = None
    raw_y = None

    for i in range(n_iter):

        sampling_args = {}
        if sample_method == lc_representative:
            sampling_args = {'raw_sent': train_sent,
                             'raw_x': raw_x,
                             'raw_y': raw_y}

        sampled_index = active_learner.add_n_new_samples(
            sample_method,
            embeddings_train,
            train_y,
            train_sent,
            **sampling_args
        )

        raw_x = active_learner.get_raw_train_sent().ravel()
        raw_y = active_learner.get_y_train().ravel()
        train_sent = np.delete(train_sent, sampled_index, 0)
        train_y = np.delete(train_y, sampled_index, 0)
        embeddings_train = np.delete(embeddings_train, sampled_index, 0)

        model.evaluate(raw_x, test_sent, raw_y, test_y)
        print(f'iter_{i}_out_of_{n_iter}')
        print(f'{dataset_name}_ {sample_method.__name__}')
        print(model.get_scores()['accuracy'][-1])

    return model.get_scores()


def run_multiple_experiments(
        active_learner: ActiveLearner,
        model: Model,
        embeddings_train: np.array,
        train_sent: np.array,
        test_sent: np.array,
        train_y: np.array,
        test_y: np.array,
        n_iter: int,
        sample_method_list: list,
        dataset_name: str
):
    for experiment in sample_method_list:
        res_dict = run_experiment(deepcopy(active_learner),
                                  deepcopy(model),
                                  np.copy(embeddings_train),
                                  np.copy(train_sent),
                                  np.copy(test_sent),
                                  np.copy(train_y),
                                  np.copy(test_y),
                                  n_iter,
                                  experiment,
                                  dataset_name)
        read_write_results(res_dict, experiment, dataset_name)

    return print('Done')


def run_experiments_with_cross_validation(
        data,
        dataset_name,
        experiments_configs: List,
        model: Model,
        n_sample: int,
        kf_splits: int = 5,
        initialization_method: Callable = random,
        embedding_weights=None

):
    kf = KFold(n_splits=kf_splits, shuffle=True)

    n_iter = ((len(data) // kf_splits) * (kf_splits-1)) // n_sample

    results = []

    for (k, (train_index, test_index)), config in product(enumerate(kf.split(data)), experiments_configs):
        representations = np.array([np.array(sent.tolist()) for sent in data[config['representation']].tolist()])
        labels = np.array([np.array([label]) for label in data.Label.values])
        sentences = np.array([sent for sent in data.sentence.tolist()])

        train_representations, test_representations = representations[train_index], representations[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        train_sentences, test_sentences = sentences[train_index], sentences[test_index]

        learner = ActiveLearner(
            initialization_method=initialization_method,
            n_samples=n_sample
        )

        res = run_experiment(deepcopy(learner),
                             deepcopy(model),
                             np.copy(train_representations),
                             np.copy(train_sentences),
                             np.copy(test_sentences),
                             np.copy(train_labels),
                             np.copy(test_labels),
                             n_iter,
                             config['sample_method'],
                             dataset_name)

        res['k_fold'] = [k] * n_iter
        res['sample_method'] = [config['sample_method'].__name__] * n_iter
        results.append(res)

    write_results(results, dataset_name)


# DATA_SET = r'experiments_data/imdb.parquet'
DATA_SET = r'data_with_vectors/amazon_cells_labelled.parquet'
dataset_name = 'amazon'
N_SAMPLE = 50
TEST_SIZE = 0.2
BATCH_SIZE = 100
LSTM = False
data = pd.read_parquet(DATA_SET)

m = Model('RandomForest')

experiment_configs = [
    # {'representation': 'SentenceBert', 'sample_method': random_sample},
    # {'representation': 'SentenceBert', 'sample_method': lc_representative},
    # {'representation': 'AvgBert', 'sample_method': random_sample},
    # {'representation': 'AvgBert', 'sample_method': lc_representative},
    {'representation': 'embedded', 'sample_method': random_sample},
    {'representation': 'embedded', 'sample_method': lc_representative}
]

run_experiments_with_cross_validation(
    data,
    dataset_name,
    experiment_configs,
    m,
    N_SAMPLE,
)
# plot_sample_method('toxic', 'f1')

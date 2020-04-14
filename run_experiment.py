import time
from itertools import product
from pathlib import PurePosixPath, Path
from typing import Callable, Dict, List
from utils import read_write_results, plot_sample_method, write_results, pivot_and_plot
from copy import deepcopy
import pandas as pd
from active_learning import ActiveLearner
from sample_methods import *
from sklearn.model_selection import train_test_split, KFold
from model import Model
from sample_methods import ADDITION_SAMPLE_ARGS, EXPERIMENT_METHODS


def run_experiment(
        active_learner: ActiveLearner,
        model_type: str,
        embeddings_train: np.array,
        unlabelled_sentences: np.array,
        test_sent: np.array,
        train_y: np.array,
        test_y: np.array,
        n_iter: int,
        sample_method: Callable,
        random_init_sample,
        dataset_name: str
) -> Dict:
    labelled_sentences = None
    labelled_sentences_labels = None

    model = Model(model_type)

    for i in range(n_iter):

        sampling_args = {}

        if sample_method in ADDITION_SAMPLE_ARGS:
            sampling_args = {'unlabelled_sentences': unlabelled_sentences,
                             'labelled_sentences': labelled_sentences,
                             'labels': labelled_sentences_labels,
                             'model_type': model_type}

        sampled_index = active_learner.add_n_new_samples(
            sample_method,
            embeddings_train,
            train_y,
            unlabelled_sentences,
            random_init_sample,
            **sampling_args
        )

        labelled_sentences = active_learner.get_raw_train_sent()
        labelled_sentences_labels = active_learner.get_y_train().ravel()

        if labelled_sentences.dtype != float:
            labelled_sentences = labelled_sentences.ravel()

        unlabelled_sentences, train_y, embeddings_train = remove_used_index(sampled_index, unlabelled_sentences,
                                                                            train_y, embeddings_train)

        model.evaluate(labelled_sentences, test_sent, labelled_sentences_labels, test_y)
        print(f'iter_{i}_out_of_{n_iter}')
        print(f'{dataset_name}_ {sample_method.__name__}')
        print(model.get_scores()['accuracy'][-1])

    return model.get_scores()


def remove_used_index(
        sampled_index: np.array,
        train_sent: np.array,
        train_y: np.array,
        embeddings_train: np.array
):
    train_sent = np.delete(train_sent, sampled_index, 0)
    train_y = np.delete(train_y, sampled_index, 0)
    embeddings_train = np.delete(embeddings_train, sampled_index, 0)
    return train_sent, train_y, embeddings_train


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
        n_sample: int,
        n_iter: int = 10,
        kf_splits: int = 5,
        initialization_method: Callable = random_sample_init,

):
    kf = KFold(n_splits=kf_splits, shuffle=True)

    n_iter = (((len(data) // kf_splits) * (kf_splits - 1)) // n_sample) - 1

    results = []
    random_samples_dic = dict()

    for (k, (train_index, test_index)), config in product(enumerate(kf.split(data)), experiments_configs):
        representations = np.array([np.array(sent.tolist()) for sent in data[config['representation']].tolist()])
        labels = np.array([np.array([label]) for label in data.Label.values])
        sentences = np.array([sent for sent in data.sentence.tolist()])

        if config['model_type'] in ORIGINAL_REPRESENTATION_MODELS:
            sentences = representations

        train_representations, test_representations = representations[train_index], representations[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        train_sentences, test_sentences = sentences[train_index], sentences[test_index]

        random_samples_dic[k] = random_samples_dic.get(k, np.random.randint(len(train_index),
                                                                            size=n_sample))  # generate n_sample random indexes from train_index.
        random_init_sample = random_samples_dic.get(k)
        learner = ActiveLearner(
            initialization_method=initialization_method,
            n_samples=n_sample
        )

        res = run_experiment(deepcopy(learner),
                             config['model_type'],
                             np.copy(train_representations),
                             np.copy(train_sentences),
                             np.copy(test_sentences),
                             np.copy(train_labels),
                             np.copy(test_labels),
                             n_iter,
                             config['sample_method'],
                             random_init_sample,
                             dataset_name)

        res['k_fold'] = [k] * n_iter
        res['sample_method'] = [config['sample_method'].__name__] * n_iter
        res['representation'] = [config['representation']] * n_iter
        res['model_type'] = [config['model_type']] * n_iter
        results.append(res)

    write_results(results, dataset_name)


if __name__ == '__main__':
    DATA_SETS = ['yelp_cells_labelled.parquet', 'amazon_cells_labelled.parquet', 'imdb_with_vectors.parquet']
    DATA_PATH = 'data_with_vectors'
    DATA_SET = 'yelp_cells_labelled.parquet'
    DATA_SET_PATH = Path(DATA_PATH) / DATA_SET
    dataset_name = str(DATA_SET_PATH).rpartition('\\')[-1].rpartition('.')[0]
    N_SAMPLE = 50
    data = pd.read_parquet(DATA_SET_PATH)

    ORIGINAL_REPRESENTATION_MODELS = ['RF', 'SVM', 'LR']
    TFIDF_REPRESENTATION_MODELS = ['RandomForest', 'SVC', 'LogisticRegression']
    REPRESENTATIONS = ['SentenceBert', 'AvgBert']
    MODELS_LIST = TFIDF_REPRESENTATION_MODELS

    experiment_configs = [
        {'representation': representation, 'sample_method': sample_method, 'model_type': model_type} for
        (sample_method, model_type, representation) in
        product(EXPERIMENT_METHODS, MODELS_LIST, REPRESENTATIONS)

    ]

    run_experiments_with_cross_validation(
        data,
        dataset_name,
        experiment_configs,
        N_SAMPLE,
    )

    timestamp = time.strftime("%d_%m_%Y_%H%M%S")
    file_name = dataset_name + timestamp + '.csv'
    df = pd.read_csv(f'results/{dataset_name}/{file_name}')
    for model_type in MODELS_LIST:
        pivot_and_plot(df, 'f1', model_type)
        pivot_and_plot(df, 'accuracy', model_type)


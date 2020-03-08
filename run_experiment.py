from typing import Callable, Dict
from utils import read_write_results

import pandas as pd
from active_learning import ActiveLearner
from sample_methods import *
from sklearn.model_selection import train_test_split
from model import Model

# TODO :
#  1. add pre process part ,
#  2. add more sample methods,
#  3. add a method to run experiments and save results,
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

    for i in range(n_iter):
        sampled_index = active_learner.add_n_new_samples(
            sample_method=sample_method,
            x=embeddings_train,
            y=train_y,
            raw_sentences=train_sent)

        train_sent = np.delete(train_sent, sampled_index, 0)
        train_y = np.delete(train_y, sampled_index, 0)
        embeddings_train = np.delete(embeddings_train, sampled_index, 0)

        model.evaluate(active_learner.get_raw_train_sent(), test_sent, active_learner.get_y_train(), test_y)
        read_write_results(model.get_scores(),sample_method,dataset_name)

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
        ) -> Dict:


    for experiment in sample_method_list:
        run_experiment(active_learner,
                       model,
                       embeddings_train,
                       train_sent,
                       test_sent,
                       train_y,
                       test_y,
                       n_iter,
                       experiment,
                       dataset_name)
    return print('Done')



DATA_SET = r'/Users/uri/nlp_active_learning/data_with_vectors/yelp_labelled.parquet'
dataset_name = 'yelp'
N_SAMPLE = 20
TEST_SIZE = 0.2

data = pd.read_parquet(DATA_SET)
embeddings = np.array([np.array(sent.tolist()) for sent in data.embedded.tolist()])
labels = np.array([np.array([label]) for label in data.Label.values])
sentences = np.array([sent for sent in data.sentence.tolist()])
indices = np.arange(len(sentences))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(embeddings, labels, indices,
                                                                                 test_size=TEST_SIZE)

train_sentences = sentences[train_indices]
test_sentences = sentences[test_indices]

# RF classifier example:
learner = ActiveLearner(
    initialization_method=group_cosine_distance_mean,
    n_samples=N_SAMPLE
)

m = Model('RandomForest')

N_ITER = int((len(embeddings) * (1 - TEST_SIZE)) // N_SAMPLE)

experiment_list = [random_sample,mdr,cosine_distance_mean]

# run experiment:

run_multiple_experiments(
        active_learner=learner,
        model=m,
        embeddings_train=X_train,
        train_sent=train_sentences,
        test_sent=test_sentences,
        train_y=y_train,
        test_y=y_test,
        n_iter=N_ITER,
        sample_method_list=experiment_list,
        dataset_name = dataset_name)

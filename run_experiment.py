from typing import Callable, Dict
from utils import read_write_results, plot_sample_method
from copy import deepcopy
import pandas as pd
from active_learning import ActiveLearner
from sample_methods import *
from sklearn.model_selection import train_test_split
from model import Model
from data_pre_process import ProcessDataset, pad_features
import torch
from torchnlp.word_to_vector import GloVe


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
        read_write_results(model.get_scores(), sample_method, dataset_name)
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
        run_experiment(deepcopy(active_learner),
                       deepcopy(model),
                       np.copy(embeddings_train),
                       np.copy(train_sent),
                       np.copy(test_sent),
                       np.copy(train_y),
                       np.copy(test_y),
                       n_iter,
                       experiment,
                       dataset_name)
    return print('Done')


DATA_SET = r'data_with_vectors/amazon_cells_labelled.parquet'
dataset_name = 'amazon'
N_SAMPLE = 100
TEST_SIZE = 0.2
BATCH_SIZE = 100
data = pd.read_parquet(DATA_SET)
processed_data = ProcessDataset(data, max_vocab_size=len(data))
words_counter = processed_data.build_counter()
vocab, index_to_vocab = processed_data.build_vocab(words_counter=words_counter, max_vocab_size=len(data))
embeddings = np.array([np.array(sent.tolist()) for sent in data.embedded.tolist()])
labels = np.array([np.array([label]) for label in data.Label.values])
# sentences = np.array([sent for sent in data.sentence.tolist()])
# sentences = np.array([pad_features(processed_data.vectorize(sent, vocab)) for sent in data.sentence.tolist()])
sentences = np.array([pad_features(processed_data.__getitem__(i)[0]) for i in range(len(data))])
indices = np.arange(len(sentences))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(embeddings, labels, indices,
                                                                                 test_size=TEST_SIZE)

pretrained_embedding = GloVe(name='6B', dim=300, is_include=lambda w: w in vocab.keys())
embedding_weights = torch.Tensor(len(vocab.keys()), pretrained_embedding.dim)

for num, word in index_to_vocab.items():
    embedding_weights[num] = pretrained_embedding[index_to_vocab[num]]

train_sentences = sentences[train_indices]
test_sentences = sentences[test_indices]
# RF classifier example:
learner = ActiveLearner(
    initialization_method=group_cosine_distance_mean,
    n_samples=N_SAMPLE
)
m = Model('LSTM',
          **{
              'label_size': len(np.unique(labels)),
              'vocab_size': len(vocab),
              'batch_size': min(BATCH_SIZE, N_SAMPLE),
              'pretrained_embeddings': embedding_weights
          })
N_ITER = int((len(embeddings) * (1 - TEST_SIZE)) // N_SAMPLE)

experiment_list = [random_sample, mdr]

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
    dataset_name=dataset_name)
plot_sample_method('amazon', 'f1')




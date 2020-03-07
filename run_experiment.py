import pandas as pd
from nlp_active_learning.active_learning import ActiveLearner
from nlp_active_learning.sample_methods import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nlp_active_learning.model import Model

# TODO :
#  1. add pre process part ,
#  2. add more sample methods,
#  3. add a method to run experiments and save results,
#  4. parallel run of multiple experiments

DATA_SET = r'data_with_vectors\imdb_labelled.parquet'
N_SAMPLE = 100
TEST_SIZE = 0.2

data = pd.read_parquet(DATA_SET)
embeddings = np.array([np.array(sent.tolist()) for sent in data.embedded.tolist()])
labels = np.array([np.array([label]) for label in data.Label.values])
sentences = np.array([sent for sent in data.sentence.tolist()])
indices = np.arange(len(sentences))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(embeddings, labels, indices, test_size=TEST_SIZE)

train_sentences = sentences[train_indices]
test_sentences = sentences[test_indices]

# RF classifier example:
learner = ActiveLearner(
    initialization_method=group_cosine_distance_mean,
    n_samples=N_SAMPLE
)

m = Model('RandomForest')

N_ITER = int((len(embeddings) * (1 - TEST_SIZE)) // N_SAMPLE)

# learning loop:
for i in range(N_ITER):

    sampled_index = learner.add_n_new_samples(sample_method=cosine_distance_mean, x=X_train, y=y_train)
    X_train = np.delete(X_train, sampled_index, 0)
    sentences_to_fit = train_sentences[sampled_index]
    y_to_fit = y_train[sampled_index]
    train_sentences = np.delete(train_sentences, sampled_index, 0)
    y_train = np.delete(y_train, sampled_index, 0)
    m.evaluate(sentences_to_fit, test_sentences, y_to_fit, y_test)


# results:
m.get_scores()

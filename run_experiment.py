import pandas as pd
from nlp_active_learning.active_learning import ActiveLearner
from nlp_active_learning.sample_methods import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# TODO :
#  1. add pre process part ,
#  2. add more sample methods,
#  3. add a method to run experiments and save results,
#  4. parallel run of multiple experiments

DATA_SET = r'data_with_vectors\imdb_labelled.parquet'
N_SAMPLE = 100
TEST_SIZE = 0.2

data = pd.read_parquet(DATA_SET)
sentences = np.array([np.array(sent.tolist()) for sent in data.embedded.tolist()])
labels = np.array([np.array([label]) for label in data.Label.values])

X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=TEST_SIZE)

# RF classifier example:
learner = ActiveLearner(
    clf=RandomForestClassifier(),
    initialization_method=group_cosine_distance_mean,
    n_samples=N_SAMPLE
)

N_ITER = int((len(sentences)*(1-TEST_SIZE)) // N_SAMPLE)

# learning loop:
for i in range(N_ITER):

    sampled_index = learner.add_n_new_samples(sample_method=cosine_distance_mean, x=X_train, y=y_train)
    X_train = np.delete(X_train, sampled_index, 0)
    y_train = np.delete(y_train, sampled_index, 0)
    learner.fit_model()
    learner.accuracy_score(X_test, y_test)
    learner.f1_score(X_test, y_test)

# results:
f1 = learner.get_f1_scores()
accuracy = learner.get_accuracy_scores()

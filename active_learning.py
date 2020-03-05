from typing import Callable
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from nlp_active_learning.sample_methods import cosine_distance_mean, group_cosine_distance_mean


class ActiveLearner:
    """
    use example:
    >>> learner = ActiveLearner(clf=Classifier, initialization_method=group_cosine_distance_mean, n_samples=100)
    >>> sampled_index = learner.add_n_new_samples(x, y , sample_method=cosine_distance_mean)
    >>> learner.fit_model()
    """

    def __init__(self, n_samples, clf, initialization_method):

        self._clf = clf
        self._initialization_method = initialization_method
        self._n_samples = n_samples
        self._train_sentences = None
        self._train_labels = None
        self._f1_score = []
        self._accuracy_score = []

    def initialize_learner(self, x: np.array, y: np.array) -> np.array:
        """
        :param x: embedded sentences
        :param y: labels
        """

        ind = self._initialization_method(x)
        self._train_sentences = x[ind]
        self._train_labels = y[ind]

        return ind

    def add_n_new_samples(
            self,
            sample_method: Callable,
            x: np.array,
            y: np.array,
            *sampling_args
    ):

        """
        :param sample_method: strategy of sampling new sentences from x
        :param sampling_args: additional args that the sample method should get
        :param x: embedded sentences
        :param y: labels
        """

        assert len(x) == len(y), "len of sentences doesn't match len of labels"
        assert len(x) >= self._n_samples, "there are not enough samples to add"

        if not self._train_sentences:
            ind = self._initialization_method(x, y)

        else:
            ind = sample_method(x, self._train_sentences, self._n_samples, *sampling_args)
            self._train_sentences = np.vstack((self._train_sentences, x[ind]))
            self._train_labels = np.vstack((self._train_labels, y[ind]))

        return ind

    def fit_model(self):
        self._clf.fit(self._train_sentences, self._train_labels)

    def predict(self, x_test: np.array):
        return self._clf.predict(x_test)

    def f1_score(
            self,
            x_test: np.array,
            y_test: np.array,
    ):
        y_pred = self.predict(x_test)
        self._f1_score.append(f1_score(y_test, y_pred))

    def accuracy_score(
            self,
            x_test: np.array,
            y_test: np.array,
    ):
        y_pred = self.predict(x_test)
        self._accuracy_score.append(accuracy_score(y_test, y_pred))

    def get_f1_scores(self):
        return self._f1_score

    def get_accuracy_scores(self):
        return self._accuracy_score










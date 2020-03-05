from typing import Callable
import numpy as np


class ActiveLearner:
    """
    use example:
    >>> learner = ActiveLearner(train_sentences, train_labels,clf=Classifier, n_samples=100)
    >>> initialized_index = learner.initialize_learner(initialization_method=random_sampling)
    >>> sampled_index = learner.add_n_new_samples(sample_method=cosine_distance)
    >>> learner.fit_model()
    >>> predictions = learner.predict()

    """

    def __init__(self, n_samples, clf):
        self._clf = clf
        self._n_samples = n_samples
        self._train_sentences = None
        self._train_labels = None

    def initialize_learner(self, initialization_method: Callable, x: np.array, y: np.array) -> np.array:
        """
        :param initialization_method: a function ,
         input x : embedded sentences, output:  index to sample from x
        :param x: embedded sentences
        :param y: labels
        """

        ind = initialization_method(x)
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
        assert (self._train_sentences & self._train_labels), \
            "train sentences and labels are None, initialize them first"
        assert len(x) == len(y), "len of sentences doesn't match len of labels"
        assert len(x) >= self._n_samples, "there are not enough samples to add"

        ind = sample_method(x, self._train_sentences, self._n_samples, *sampling_args)

        self._train_sentences = np.vstack(self._train_sentences, x[ind])
        self._train_labels = np.vstack(self._train_labels, y[ind])
        return ind

    def fit_model(self):
        self._clf.fit(self._train_sentences, self._train_labels)

    def predict(self, x_test):
        return self._clf.predict(x_test)









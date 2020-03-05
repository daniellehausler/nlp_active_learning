from typing import Callable
import numpy as np


class ActiveLearner:
    """
    use example:
    >>> learner = ActiveLearner(train_sentences, train_labels,clf=Classifier, n_samples=100)
    >>> learner.initialize_learner(initialization_method=random_sampling)
    >>> learner.add_n_new_samples(sample_method=cousin_distance)
    >>> learner.fit_model()
    >>> predictions = learner.predict()

    """

    def __init__(self, n_samples, x, y, clf):
        self._x = x
        self._y = y
        self._clf = clf
        self._n_samples = n_samples
        self._train_sentences = None
        self._train_labels = None

    def initialize_learner(self, initialization_method: Callable) -> np.array:
        """

        :param initialization_method: a function ,
         input x : embedded sentences, output:  index to sample from x
        """

        ind = initialization_method(self._x)
        self._train_sentences = self._x[ind]
        self._train_labels = self._y[ind]
        self.drop_indices_from_array(self._x, ind),
        self.drop_indices_from_array(self._y, ind)

    def add_n_new_samples(
            self,
            sample_method: Callable,
            *sampling_args
    ):

        """
        :param sample_method: strategy of sampling new sentences from x
        :param sampling_args: additional args that the sample method should get
        """
        assert (self._train_sentences & self._train_labels), \
            "train sentences and labels are None, initialize them first"
        assert len(self._x) == len(self._y), "len of sentences doesn't match len of labels"
        assert len(self._x) >= self._n_samples, "there are not enough samples to add"

        ind = sample_method(self._x, *sampling_args)

        self._train_sentences = np.vstack(self._train_sentences, self._x[ind])
        self._train_labels = np.vstack(self._train_labels, self._y[ind])

        self.drop_indices_from_array(self._x, ind),
        self.drop_indices_from_array(self._y, ind)

    @staticmethod
    def drop_indices_from_array(a, index):
        return np.delete(a, index)

    def fit_model(self):
        self._clf.fit(self._train_sentences, self._train_labels)

    def predict(self, x_test):
        return self._clf.predict(x_test)









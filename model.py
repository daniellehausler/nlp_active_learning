from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
import torch.nn as nn
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import TensorDataset, DataLoader


class Model:
    """
    >>> m = Model('RandomForest')
    >>> m.evaluate(X_train, X_test, y_train, y_test)
    >>> m.get_scores()
    """

    def __init__(self, clf_name, **kwargs):
        # TODO : Option for other type of solvers (lstm etc)
        self._clf = {
            'RandomForest': self.sklearn_pipeline(RandomForestClassifier()),
            'RBF_SVM': self.sklearn_pipeline(RandomForestClassifier(SVC(gamma=2, C=1))),
            'LSTM': LSTM(n_epochs=5, model=LSTMClassifier(**kwargs))
        }
        self._model = self._clf[clf_name]
        self._scores = {'accuracy': [],
                        'f1': [], 'n_samples': []}
        self._kwargs = kwargs

    @staticmethod
    def sklearn_pipeline(clf):
        sklearn_pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer()),
                ("classifier", clf),
            ]
        )
        return sklearn_pipeline

    def fit(self, train_sentences, y_train):
        self._model.fit(train_sentences, y_train)

    def predict(self, test_sentences):
        return self._model.predict(test_sentences)

    def proba(self, test_sentences):
        return self._model.predict_proba(test_sentences)

    def log_proba(self, test_sentences):
        return self._model.predict_log_proba(test_sentences)

    def accuracy(self, y_test, y_pred):
        self._scores['accuracy'].append(accuracy_score(y_test, y_pred))

    def f1(self, y_test, y_pred):
        self._scores['f1'].append(f1_score(y_test, y_pred))

    def count_samples(self, x_train):
        self._scores['n_samples'].append(len(x_train))

    def evaluate(self, train_sentences, test_sentences, y_train, y_test):
        self.fit(train_sentences, y_train)
        y_pred = self.predict(test_sentences)
        self.count_samples(train_sentences)
        self.f1(y_test.reshape(len(y_test),), y_pred)
        self.accuracy(y_test, y_pred)

    def get_scores(self):
        return self._scores


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size=0, label_size=0, batch_size=0, pretrained_embeddings=None, embedding_dim=300, hidden_dim=300):
        super(LSTMClassifier, self).__init__()
        self._hidden_dim = hidden_dim
        self._batch_size = batch_size
        if pretrained_embeddings is not None:
            self._word_embeddings = nn.Embedding(vocab_size, embedding_dim).from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        else:
            self._word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self._lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self._hidden2label = nn.Linear(hidden_dim, label_size)
        self._hidden = self.init_hidden()

    def init_hidden(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        h0 = torch.zeros(1, self._batch_size, self._hidden_dim).zero_().to(device)
        c0 = torch.zeros(1, self._batch_size, self._hidden_dim).zero_().to(device)
        return (h0, c0)

    def forward(self, sentences):
        embeddings = self._word_embeddings(sentences)
        lstm_out, self._hidden = self._lstm(embeddings, self._hidden)
        y = self._hidden2label(lstm_out.reshape(-1, len(sentences), embeddings.shape[2])[-1])
        log_probs = F.log_softmax(y, dim=1)
        return log_probs


class LSTM:
    def __init__(self,
                 n_epochs,
                 model,
                 criterion=torch.nn.CrossEntropyLoss,
                 optimizer=torch.optim.Adam
                 ):
        self._model = model
        self._n_epochs = n_epochs
        self._batch_size = self._model._batch_size
        self._criterion = criterion
        self._optimizer = optimizer(params=self._model.parameters(), lr=0.01)

    def train(self, sentences, labels):
        model = self._model

        if self._model._batch_size != len(sentences):
            self._model._batch_size = len(sentences)
            self._model._hidden = self._model.init_hidden()

        label_probs = model(sentences)
        model.zero_grad()
        self._model._hidden = self._model.init_hidden()
        loss_func = self._criterion()
        loss = loss_func(label_probs, labels.reshape(len(labels), ))
        loss.backward()
        nn.utils.clip_grad_norm_(self._model.parameters(), 5)
        self._optimizer.step()
        return label_probs, loss

    def fit(self, sentences, labels):
        train_data = TensorDataset(torch.LongTensor(sentences), torch.LongTensor(labels.reshape(len(labels), -1)))
        train_loader = DataLoader(train_data, batch_size=self._batch_size, shuffle=True)
        step = 0
        for epoch in range(self._n_epochs):
            for inputs, labels in train_loader:
                step += 1
                label_probs, loss = self.train(inputs, labels)
                print(
                    f"Epoch: {epoch + 1}/{self._n_epochs} \n",
                    f"Step: {step} \n",
                    f"Training Loss: {loss:.4f}"
                )
                self._model.train()

    def predict(self, test_sentences):
        y_pred = []
        self._model.eval()
        test_data = TensorDataset(torch.LongTensor(test_sentences))
        test_loader = DataLoader(test_data, batch_size=self._batch_size, shuffle=True)
        for inputs in test_loader:
            if self._model._batch_size != len(inputs[0]):
                self._model._batch_size = len(inputs[0])
                self._model._hidden = self._model.init_hidden()

            label_probs = self._model(inputs[0])
            y_pred = y_pred + [torch.argmax(prob) for prob in label_probs]
        return torch.IntTensor(y_pred)

    def predict_proba(self, test_sentences):
        y_pred = []
        self._model.eval()
        test_data = TensorDataset(torch.LongTensor(test_sentences))
        test_loader = DataLoader(test_data, batch_size=self._batch_size, shuffle=True)
        for inputs in test_loader:
            log_probs = self._model(inputs[0])
            y_pred.append(torch.exp(log_probs))
        return y_pred

    def predict_log_proba(self, test_sentences):
        y_pred = []
        self._model.eval()
        test_data = TensorDataset(torch.LongTensor(test_sentences))
        test_loader = DataLoader(test_data, batch_size=self._batch_size, shuffle=True)
        for inputs in test_loader:
            log_probs = self._model(inputs[0])
            y_pred.append(log_probs)
        return y_pred

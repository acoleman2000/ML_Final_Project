from abc import ABC, abstractmethod # Abstract Base Classes for Python
import tensorflow as tf
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest

#@title Define a base class class for Learning Algorithm

class BaseLearningAlgorithm(ABC):
  """Base class for a Supervised Learning Algorithm."""

  @abstractmethod
  def fit(self, ds_train, ds_val):
    """Trains the classifier dataset with a train and validation dataset."""

  @abstractmethod
  def predict(self, x_test):
    """Predicts one or more examples in the dataset.."""

  @property
  @abstractmethod
  def name(self) -> str:
    """Returns the name of the algorithm."""

  @property
  def layers(self):
    """Returns the layers in the model as a list."""
    raise NotImplementedError()

  @property
  def input(self):
    """Returns the activations per layer, used for displaying activations."""
    raise NotImplementedError()


class RF(BaseLearningAlgorithm):
    def __init__(self, criterion, n_estimators, max_depth, max_features):
        self._n_estimators = n_estimators
        self._criterion = criterion
        self._max_depth = max_depth
        self._max_features = max_features
        self._model = RandomForestClassifier(criterion = criterion, n_estimators = n_estimators, max_depth = max_depth, max_features = max_features, random_state = 18)

    def fit(self, ds_train, ds_val):
        X_train = ds_train[:,0:-2]
        y_train = ds_train[:,-1]
        self._model.fit(X_train, y_train)

    def predict(self, ds_test):
        X_test = ds_test[:,0:-2]
        return self._model.predict(X_test)

    def name(self):
        return "Random Forest Classifier"

class IF(BaseLearningAlgorithm):
    def __init__(self, max_samples, n_estimators, contamination, bootstrap):
        self._max_samples = max_samples
        self._n_estimators = n_estimators
        self._contamination = contamination
        self._bootstrap = bootstrap
        self._model = IsolationForest(max_samples= max_samples, n_estimators=n_estimators,contamination=contamination, bootstrap=bootstrap)

    def fit(self, ds_train, ds_val):
        X_train = ds_train[:,0:-2]
        self._model.fit(X_train)


    def predict(self, ds_test):
        X_test = ds_test[:,0:-2]
        f = lambda x: 0 if x == 1 else 1
        prediction = self._model.predict(X_test)
        return list(map(f, prediction))

    def name(self):
        return "Isolation Forest Classifier"


class OCSVM(BaseLearningAlgorithm):
    def __init__(self, kernel, gamma, tol):
        self._kernel = kernel
        self._gamma = gamma
        self._tol = tol
        self._model = OneClassSVM(kernel=kernel, gamma=gamma, tol=tol)

    def fit(self, ds_train, ds_val):
        X_train = ds_train[:,0:-2]
        self._model.fit(X_train)


    def predict(self, ds_test):
        X_test = ds_test[:,0:-2]
        f = lambda x: 0 if x == 1 else 1
        prediction = self._model.predict(X_test)
        return list(map(f, prediction))

    def name(self):
        return "One Class SVM"


class CNN(BaseLearningAlgorithm):
    def __init__(self, convolutional_layers, dense_layers, optimizer, epochs, steps_per_epochs, validation_steps):
        self._convolutional_layers = convolutional_layers
        self._dense_layers = dense_layers
        self._optimizer = optimizer
        self._epochs = epochs
        self._steps_per_epoch = steps_per_epochs
        self._validation_steps = validation_steps

        layers = []
        for layer in convolutional_layers:
            layers.append(layer)

        layers.append(tf.keras.layers.Flatten())

        for layer in dense_layers:
            layers.append(layer)

        layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
        self._model = tf.keras.models.Sequential(layers)
        self._model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])


    def fit(self, ds_train, ds_val):
        history = self._model.fit(ds_train,
            steps_per_epoch=self._steps_per_epoch,
            epochs=self._epochs,
            verbose=1,
            validation_data = ds_val,
            validation_steps=self._validation_steps)

    def predict(self, ds_test):
        X_test = ds_test[:,0:-2]
        return self._model.predict(X_test)
    def name(self):
        print("Convolutional Neural Network")

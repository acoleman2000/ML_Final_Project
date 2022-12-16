from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.image as mpimg
import seaborn as sns; sns.set()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import h5py
import os

def train_eval(learning_algo, ds_train, ds_val, ds_test):
  """Trains and evaluates the generic model."""

  learning_algo.fit(ds_train, ds_val)



  y_pred = learning_algo.predict(ds_test)
  if type(ds_test) == np.ndarray:
    y_test = ds_test[:,-1]
  else:
    y_test = ds_test.labels
  mat = confusion_matrix(y_test, y_pred)
  sns.set(rc = {'figure.figsize':(8,8)})
  sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
              xticklabels=['%d' %i for i in range(2)],
              yticklabels=['%d' %i for i in range(2)])
  plt.xlabel('true label')
  plt.ylabel('predicted label')
  plt.title(learning_algo.name())

  print(classification_report(y_test, y_pred,
                              target_names=['%d' %i for i in range(2)]))


  # fpr = dict()
  # tpr = dict()
  # roc_auc = dict()
  # for i in range(n_classes):
  #   fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
  #   roc_auc[i] = auc(fpr[i], tpr[i])


  #   plt.figure()
  # lw = 2
  # plt.plot(
  #   fpr[2],
  #   tpr[2],
  #   color="darkorange",
  #   lw=lw,
  #   label="ROC curve (area = %0.2f)" % roc_auc[2],
  # )
  # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
  # plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.05])
  # plt.xlabel("False Positive Rate")
  # plt.ylabel("True Positive Rate")
  # plt.title("Receiver operating characteristic example")
  # plt.legend(loc="lower right")
  # plt.show()



def get_generators():

    train_datagen = ImageDataGenerator(rescale=1/255)
    validation_datagen = ImageDataGenerator(rescale=1/255)
    test_datagen = ImageDataGenerator(rescale=1/255)

    train_generator = train_datagen.flow_from_directory(
            'data/training/',
            classes = ['interesting_waterfall', 'uninteresting_waterfall'],
            target_size=(200, 200),
            batch_size=120,
            class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
            'data/validation/',
            classes = ['interesting_waterfall', 'uninteresting_waterfall'],
            target_size=(200, 200),
            batch_size=80,
            class_mode='binary',
            shuffle=False)


    test_generator = test_datagen.flow_from_directory(
            'data/test/',
            classes = ['interesting_waterfall', 'uninteresting_waterfall'],
            target_size=(200, 200),
            batch_size=120,
            class_mode='binary')

    return train_generator, validation_generator, test_generator


def gather_files(path, mode, data, y_value, start_index):
    path_files = os.listdir(path)
    index = start_index
    for file in path_files:
        data[index] = np.append(get_differences(path + file, mode), y_value)
        index+=1
    return np.array(data)

def gather_off_files(path,mode,data,y_value,start_index):
    path_files = os.listdir(path)
    index = start_index
    for file in path_files:
        data[index] = np.append(gather_values(path + file, "Off", mode), y_value)
        index+=1
    return np.array(data)

def gather_values(path, cadence, mode):
    values = np.array([])
    for i in range(1, 4):
        filename = "%s/frame_%s_%s.h5"%(path,cadence,i)
        with h5py.File(filename, "r") as f:
            a_group_key = list(f.keys())[0]
            ds_arr = tranform_observation(f[a_group_key][()])
            if mode == "append":
                values = np.concatenate(values, ds_arr)
            if mode == "average":
                if (values.shape[0] == 0):
                    values = ds_arr
                else:
                    values += ds_arr
    if mode == "average":
        values /= 3
    return values


def get_differences(path, mode):
    off_values = gather_values(path, "Off", mode)
    on_values = gather_values(path, "On", mode)
    return on_values - off_values

def tranform_observation(ds_arr):
    return sum(ds_arr[0:ds_arr.shape[0]])/ds_arr.shape[0]
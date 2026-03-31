import tensorflow as tf
from tensorflow import keras
import time
tf.get_logger().setLevel(40) # suppress deprecation messages
# tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, GlobalAveragePooling1D, BatchNormalization, Conv2D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import function
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import DistanceMetric

print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False



def label_encoder(training_labels, testing_labels):
    le = preprocessing.LabelEncoder()
    le.fit(np.concatenate((training_labels, testing_labels), axis=0))
    y_train = le.transform(training_labels)
    y_test = le.transform(testing_labels)

    return y_train, y_test

class Classifier_FCN:

    def __init__(self, output_directory, input_shape, nb_classes, dataset_name, verbose=True, build=True):
        self.output_directory = output_directory

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights('/home/dmlab_a/Peiyu0/test2/fcn_weights/' + str(dataset_name) + '_model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = '/home/dmlab_a/Peiyu0/test2/fcn_weights/' + str(dataset_name) + '_best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train):

        batch_size = 16
        nb_epochs = 2000

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))
        x_train, self.x_val, y_train, self.y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

        hist = self.model.fit(
            x_train, y_train,
            batch_size=mini_batch_size,
            epochs=nb_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
            validation_data=(self.x_val, self.y_val)  # Add this line
        )

        self.model.save('/home/dmlab_a/Peiyu0/test2/fcn_weights/' + str(dataset_name) + '_last_model.hdf5')

        model = keras.models.load_model('/home/dmlab_a/Peiyu0/test2/fcn_weights/' + str(dataset_name) + '_best_model.hdf5')

    def predict(self, x_test):
        model_path = '/home/dmlab_a/Peiyu0/test2/fcn_weights/' + str(dataset_name) + '_best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred

def z_score_normalize(X):
    """Per-sample z-score normalization: (x - mean) / std"""
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True) + 1e-8  # avoid divide by 0
    return (X - mean) / std

def _resolve_ucr_root():
    env_root = os.environ.get("UCR_DATA_ROOT")
    candidates = [env_root] if env_root else []

    current = os.path.abspath(os.path.dirname(__file__))
    for _ in range(8):
        candidates.append(os.path.join(current, "UCRArchive_2018"))
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    candidates.append("/UCRArchive_2018")
    for root in candidates:
        if root and os.path.isdir(root):
            return root
    raise FileNotFoundError(
        "Could not locate 'UCRArchive_2018'. Put it at repo root or set UCR_DATA_ROOT."
    )

def readUCR(ds_name: str):
    """Load UCR time series dataset."""
    path = _resolve_ucr_root()

    train_data = np.loadtxt(os.path.join(path, ds_name, f"{ds_name}_TRAIN.tsv"), delimiter='\t')
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]

    test_data = np.loadtxt(os.path.join(path, ds_name, f"{ds_name}_TEST.tsv"), delimiter='\t')
    x_test = test_data[:, 1:]
    y_test = test_data[:, 0]

    # Apply z-score normalization for Chinatown dataset
    lists = ["GunPointAgeSpan", "GunPointMaleVersusFemale", "GunPointOldVersusYoung", "HandOutlines", "Chinatown"]
    if ds_name in lists :
        print("Applying z-score normalization for dataset ", ds_name)
        x_train = z_score_normalize(x_train)
        x_test = z_score_normalize(x_test)

    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return x_train, y_train, x_test, y_test


for dataset in [
            'GunPointAgeSpan',
            'GunPointMaleVersusFemale',
            'GunPointOldVersusYoung',
            'HandOutlines',
            'Chinatown',
            # 'FordB',
            # 'FordA',
            'FreezerRegularTrain',
            # 'Wafer'
]:

   X_train, y_train, X_test, y_test = readUCR(str(dataset))
   y_train, y_test = label_encoder(y_train, y_test)

   input_shape = X_train.shape[1:]
   nb_classes = len(np.unique(np.concatenate([y_train,y_test])))
   one_hot = to_categorical(y_train)
   dataset_name = str(dataset)

   fcn = Classifier_FCN(output_directory=os.getcwd(), input_shape=input_shape, nb_classes=nb_classes, dataset_name=dataset_name, verbose=True)
   fcn.build_model(input_shape=input_shape, nb_classes=nb_classes)
   fcn.fit(X_train, to_categorical(y_train))
   fcn.predict(X_test)

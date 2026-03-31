# FCN model
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time 
import os

from .utils.utils import calculate_metrics
from .utils.utils import save_test_duration

class Classifier_FCN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False,
                 build=True, load_weights=False, weights_directory=''):
        self.output_directory = output_directory
        self.weights_directory = weights_directory

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            if load_weights == True:
                self.model.load_weights(os.path.join(self.weights_directory, 'best_model.hdf5'))
                # self.model.load_weights(self.output_directory
                #                         .replace('resnet_augment', 'resnet')
                #                         .replace('TSC_itr_augment_x_10', 'TSC_itr_10')
                #                         + '/model_init.hdf5')
            else:
                self.model.save_weights(os.path.join(self.output_directory, 'model_init.hdf5'))
        return

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
            metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = os.path.join(self.output_directory,'best_model.hdf5')
        print(file_path)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)
    
        self.callbacks = [reduce_lr, model_checkpoint]

        return model 

    def fit(self, x_train, y_train, nb_epochs=2000):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
		# x_val and y_val are only used to monitor the test loss and NOT for training  
        batch_size = 64
        # nb_epochs = 2000

        mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

        start_time = time.time() 

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, 
                              validation_split = 0.2,  callbacks=self.callbacks)
        
        duration = time.time() - start_time

        self.model.save(self.output_directory+'last_model.hdf5')
   
#         y_pred = model.predict(x_val)

# 		# convert the predicted from binary to integer 
#         y_pred = np.argmax(y_pred , axis=1)

#         save_logs(self.output_directory, hist, y_pred, y_true, duration)

#         keras.backend.clear_session()

    def my_predict(self, x_test):
        model_path = os.path.join(self.output_directory, 'best_model.hdf5')
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        return np.argmax(y_pred, axis=1)

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        model_path = os.path.join(self.output_directory, 'best_model.hdf5')
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred

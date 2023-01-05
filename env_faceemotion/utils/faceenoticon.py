# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import CSVLogger


class FaceEmotion:

    def __init__(self):
        self.data = pd.read_csv(os.path.join('data','icml_face_data.csv'))
        self.data.columns = ['emotion', 'Usage', 'pixels']
        self.emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        self.full_train_images, self.full_train_labels = self.prepare_data(self.data[self.data['Usage']=='Training'])
        self.test_images, self.test_labels = self.prepare_data(self.data[self.data['Usage']!='Training'])
        self.train_images, self.valid_images, self.train_labels, self.valid_labels = self.split_train_test()
        self.cnn_architecture()

    def prepare_data(self, dataframe):
        self.image_array = np.zeros(shape=(len(self.data), 48, 48, 1))
        self.image_label = np.array(list(map(int, self.data['emotion'])))

        for i, row in enumerate(dataframe.index):
            self.image = np.fromstring(dataframe.loc[row, 'pixels'], dtype=int, sep=' ')
            self.image = np.reshape(self.image, (48, 48)) 
            self.image_array[i, :, :, 0] = self.image / 255

        return self.image_array, self.image_label

    
    def plot_all_emotions(self):
        self.N_train = self.train_labels.shape[0]

        self.sample_select = np.random.choice(range(self.N_train), replace=False, size=16)

        self.X_sel = self.train_images[self.sample_select, :, :, :]
        self.y_sel = self.train_labels[self.sample_select]

        plt.figure(figsize=[12, 12])
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.imshow(self.X_sel[i, :, :, 0], cmap='gray')
            plt.title(self.emotions[self.y_sel[i]])
            plt.axis('off')
        plt.show()

    
    def split_train_test(self):

        self.train_images, self.valid_images, self.train_labels, self.valid_labels = train_test_split(
            self.full_train_images,
            self.full_train_labels,
            test_size=0.2
            )

        return self.train_images, self.valid_images, self.train_labels, self.valid_labels

    
    def cnn_architecture(self):

        self.cnn = Sequential()
        # 1st Layer
        self.cnn.add(Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same', input_shape=(48, 48, 1)))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.cnn.add(Dropout(0.25))

        # 2nd Layer
        self.cnn.add(Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same'))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.cnn.add(Dropout(0.25))

        # 3td Layer
        self.cnn.add(Conv2D(256, kernel_size=(1, 1), activation='relu', padding='same'))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.cnn.add(Dropout(0.25))

        # 4th Layer
        self.cnn.add(Dense(512, activation='relu'))
        self.cnn.add(Dropout(0.5))

        # 5th Layer
        self.cnn.add(Flatten())
        self.cnn.add(Dense(7, activation='softmax'))

        print(self.cnn.summary())

        self.csv_logger = CSVLogger(os.path.join('log','training.log'), separator=',', append=False)
        self.optimizer = Adam(learning_rate=0.001)
        self.cnn.compile(loss='sparse_categorical_crossentropy',
                         optimizer=self.optimizer,
                         metrics=['accuracy'])
        
    
    def cnn_fit(self, batch_size, epochs):

        self.cnn.fit(
            self.train_images,
            self.train_labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1, 
            validation_data =(self.valid_images, self.valid_labels),
            callbacks=[self.csv_logger]
            )
                
        self.cnn_string = self.cnn.to_json()
        open(os.path.join('models', 'Face_model_architecture.json'), 'w').write(self.cnn_string)
        self.cnn.save_weights(os.path.join('models', 'Face_model_weights.h5'))


    def plot_metrics(self):

        self.log_data = pd.read_csv(os.path.join('log','training.log'), sep=',', engine='python')
        plt.plot(self.log_data['accuracy'].values)
        plt.plot(self.log_data['val_accuracy'].values)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(self.log_data['loss'].values)
        plt.plot(self.log_data['val_loss'].values)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()



    def plot_confusion_matrix(self):

        self.test_prob = self.cnn.predict(self.test_images)
        self.test_pred = np.argmax(self.test_prob, axis=1)
        self.test_accuracy = np.mean(self.test_pred == self.test_labels)
        print(self.test_accuracy)

        self.conf_mat = confusion_matrix(self.test_labels, self.test_pred)

        pd.DataFrame(
            self.conf_mat,
            columns=self.emotions.values(),
            index=self.emotions.values()
        )

        fig, ax = plot_confusion_matrix(conf_mat=self.conf_mat,
                                show_normed=True,
                                show_absolute=False,
                                class_names=self.emotions.values(),
                                figsize=(8, 8))
        fig.show()


if __name__ == "__main__":
    face = FaceEmotion()
    # face.plot_all_emotions()
    face.cnn_fit(batch_size=32, epochs=50)
    face.plot_metrics()
    face.plot_confusion_matrix()
    

import os
import time

import sklearn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras as keras
from csv import writer

# train = pd.read_csv("/Users/zhiruzhu/Desktop/data_station/fl_test/income/train.csv")
train = pd.read_csv("/home/zhiru_uchicago_edu/federated_learning_test/income/train.csv")

def preprocess(data):
    # remove space
    data.columns = [cols.replace(' ', '') for cols in data.columns]
    data["education"] = [cols.replace(' ', '') for cols in data["education"]]
    data["marital-status"] = [cols.replace(' ', '') for cols in data["marital-status"]]
    data["relationship"] = [cols.replace(' ', '') for cols in data["relationship"]]
    data["race"] = [cols.replace(' ', '') for cols in data["race"]]
    data["gender"] = [cols.replace(' ', '') for cols in data["gender"]]

    # missing data
    data = data.replace('?', np.nan)
    data.dropna(inplace=True, axis=0)

    # categorical value
    cat_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender',
                   'native-country']
    df_dummy = pd.get_dummies(data, columns=cat_columns)
    return df_dummy


train = preprocess(train)
# test = preprocess(test)

X = train.drop("income_>50K", axis=1)
y = train["income_>50K"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

scaler = sklearn.preprocessing.StandardScaler()
X_train = X_train.to_numpy(dtype=np.float32)
X_train = scaler.fit_transform(X_train)
X_test = X_test.to_numpy(dtype=np.float32)
X_test = scaler.fit_transform(X_test)

y_train = y_train.to_numpy(dtype=int)
y_test = y_test.to_numpy(dtype=int)

model = keras.Sequential([
    keras.layers.Dense(1, input_dim=X_train.shape[1], activation='sigmoid')
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

class EvaluateEpochEnd(tf.keras.callbacks.Callback):
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs={}):
        global prev_epoch_time
        scores = self.model.evaluate(x=self.X_test, y=self.y_test, verbose=0)
        print('\nTesting loss: {}, accuracy: {}\n'.format(scores[0], scores[1]))
        cur_epoch_time = time.time()
        with open("income_keras_base.csv", 'a+') as f:
            writer_object = writer(f)
            writer_object.writerow([scores[0], scores[1], cur_epoch_time-prev_epoch_time])
        prev_epoch_time = cur_epoch_time

callbacks = [
    EvaluateEpochEnd(X_test, y_test)
]

prev_epoch_time = time.time()

model.fit(
    x=X_train,
    y=y_train,
    epochs=300,
    batch_size=X_train.shape[0],
    # validation_data=(X_test, y_test),
    callbacks=[callbacks]
)

model.evaluate(x=X_test, y=y_test)
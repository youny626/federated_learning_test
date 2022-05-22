#Installing collected packages: wrapt, termcolor, tensorflow-io-gcs-filesystem, tensorflow-estimator, opt-einsum, libclang, keras-preprocessing, keras, h5py, google-pasta, gast, flatbuffers, astunparse, tensorflow
#Successfully installed astunparse-1.6.3 flatbuffers-2.0 gast-0.4.0 google-pasta-0.2.0 h5py-3.6.0 keras-2.7.0 keras-preprocessing-1.1.2 libclang-14.0.1 opt-einsum-3.3.0 tensorflow-2.7.0 tensorflow-estimator-2.7.0 tensorflow-io-gcs-filesystem-0.26.0 termcolor-1.1.0 wrapt-1.14.1

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import PIL.Image as Image

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import openfl.native as fx
from openfl.federated import FederatedModel,FederatedDataSet

log_file = "/home/cc/.local/workspace/covid.log"
if os.path.exists(log_file):
    os.remove(log_file)

input_dir = "/home/cc/federated_learning_test/covid/"

#Setup default workspace, logging, etc.
fx.init('keras_cnn_mnist', log_level='METRIC', log_file="./covid.log")

# num_samples = 1000
train_size = 0.9

# Initial overhead array
overhead = []

train_df = pd.read_csv(input_dir + "train.txt", sep=" ", header=None)
train_df.columns = ['patient id', 'filename', 'class', 'data source']
train_df = train_df.drop(['patient id', 'data source'], axis=1)

# train_df = train_df.sample(n=num_samples, random_state=0)

test_df = pd.read_csv(input_dir + "test.txt", sep=" ", header=None)
test_df.columns = ['id', 'filename', 'class', 'data source']
test_df = test_df.drop(['id', 'data source'], axis=1)

train_path = input_dir + "train/"
test_path = input_dir + "test/"

train_df, valid_df = train_test_split(train_df, train_size=train_size, random_state=0)

print(f"Negative and positive values of train: {train_df['class'].value_counts()}")
print(f"Negative and positive values of validation: {valid_df['class'].value_counts()}")
print(f"Negative and positive values of test: {test_df['class'].value_counts()}")

# Now we create the train_data and train_label that will be used for ImageDataGenerator.flow
train_data = list()
train_label = list()

# valid_data = list()
# valid_label = list()

test_data = list()
test_label = list()

for _, row in train_df.iterrows():
    file_path = train_path + row["filename"]
    cur_image = Image.open(file_path).convert('RGB')
    image_resized = cur_image.resize((200, 200))
    img_data = np.array(image_resized)
    train_data.append(img_data)
    if row["class"] == "positive":
        train_label.append(1)
    else:
        train_label.append(0)

for _, row in test_df.iterrows():
    file_path = test_path + row["filename"]
    cur_image = Image.open(file_path).convert('RGB')
    image_resized = cur_image.resize((200, 200))
    img_data = np.array(image_resized)
    test_data.append(img_data)
    if row["class"] == "positive":
        test_label.append(1)
    else:
        test_label.append(0)

train_data = np.asarray(train_data).reshape(len(train_df), 200, 200, 3)
print(train_data.shape)

# valid_data = np.asarray(valid_data).reshape(num_samples-int(num_samples*train_size), 200, 200, 3)
# print(valid_data.shape)

test_data = np.asarray(test_data).reshape(400, 200, 200, 3)
print(test_data.shape)

train_datagen = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0/255.)

# set batch size to be the entire train/test set size so I can get all data in one iteration
X_train, y_train = train_datagen.flow(train_data, train_label, batch_size=train_data.shape[0]).next()
# valid_gen = test_datagen.flow(valid_data, valid_label, batch_size=64)
X_test, y_test = test_datagen.flow(test_data, test_label, batch_size=test_data.shape[0]).next()

batch_size = 64
fl_data = FederatedDataSet(X_train, y_train, X_test, y_test, batch_size=batch_size)#, num_classes=classes)


def build_model(feature_shape, classes):

    base_model = tf.keras.applications.ResNet50V2(weights='imagenet',
                                                  input_shape=feature_shape,
                                                  include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid') # binary class
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

#Create a federated model using the build model function and dataset
fl_model = FederatedModel(build_model, data_loader=fl_data)

num_collaborators = 8
collaborator_models = fl_model.setup(num_collaborators=num_collaborators)
collaborators = {}
for i in range(num_collaborators):
    collaborators[i] = collaborator_models[i]

print(f'Original training data size: {len(X_train)}')
print(f'Original validation data size: {len(X_test)}\n')

for i, model in enumerate(collaborator_models):
    print(f'Collaborator {i}\'s training data size: {len(model.data_loader.X_train)}')
    print(f'Collaborator {i}\'s validation data size: {len(model.data_loader.X_valid)}\n')

# Run experiment, return trained FederatedModel
final_fl_model = fx.run_experiment(collaborators,
                                   override_config={
        'aggregator.settings.rounds_to_train': 30,
        # 'aggregator.settings.log_metric_callback': write_metric_x,
        # "aggregator.settings.write_logs": True,
    }
)

final_fl_model.save_native('final_covid_model')
import os

import openfl.native as fx
import sklearn
from openfl.federated import FederatedModel, FederatedDataSet

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras as keras

log_file = "/home/zhiru_uchicago_edu/.local/workspace/logs/income_keras_openfl.log"
# log_file = "/Users/zhiruzhu/.local/workspace/logs/income.log"
if os.path.exists(log_file):
    os.remove(log_file)

fx.init('keras_cnn_mnist', log_level='METRIC', log_file='./logs/income_keras_openfl.log')

# train = pd.read_csv("/Users/zhiruzhu/Desktop/data_station/fl_test/income/train.csv")
train = pd.read_csv("/home/zhiru_uchicago_edu/federated_learning_test/income/train.csv")

# test = pd.read_csv("/Users/zhiruzhu/Desktop/data_station/fl_test/income/test.csv")

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

train = pd.concat([train]*10)
train.reset_index()

X = train.drop("income_>50K", axis=1)
y = train["income_>50K"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

scaler=sklearn.preprocessing.StandardScaler()
X_train = X_train.to_numpy(dtype=np.float32)
X_train = scaler.fit_transform(X_train)
X_test = X_test.to_numpy(dtype=np.float32)
X_test = scaler.fit_transform(X_test)

y_train = y_train.to_numpy(dtype=int)
y_test = y_test.to_numpy(dtype=int)

def one_hot(labels, classes):
    return np.eye(classes)[labels].astype(int)

# y_train = one_hot(y_train, 2)
# y_test = one_hot(y_test, 2)

# X_train=torch.from_numpy(X_train.astype(np.float32))
# X_test=torch.from_numpy(X_test.astype(np.float32))
# y_train=torch.from_numpy(y_train.astype(int))
# y_test=torch.from_numpy(y_test.astype(int))

# y_train=y_train.view(y_train.shape[0],1)
# y_test=y_test.view(y_test.shape[0],1)
#
# print(y_train)
# print(y_test)

# batch_size = 32
num_classes = 2

# print(X_train.shape[0])

fl_data = FederatedDataSet(X_train, y_train, X_test, y_test, batch_size=X_train.shape[0], num_classes=num_classes)

input_dim = int(X_train.shape[1])
# print(input_dim)
output_dim = 1


def build_model(feature_shape, classes):

    # number_of_classes = 2
    # number_of_features = X_train.shape[1]

    # model = tf.keras.Sequential()
    # model.add(tf.keras.Dense(number_of_classes, activation='sigmoid', input_dim=number_of_features))
    model = keras.Sequential([
        keras.layers.Dense(1, input_shape=feature_shape, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

import time

def write_metric_x(node_name, task_name, metric_name, metric, round_number):
    with open("./logs/income_openfl.log", "a") as log:
        log.write("{}/{}/{}: {}, {}, {}\n".format(node_name, task_name, metric_name, metric, round_number, time.time()))

# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('./logs/income', flush_secs=5)

# def write_metric(node_name, task_name, metric_name, metric, round_number):
#     writer.add_scalar("{}/{}/{}".format(node_name, task_name, metric_name),
#                       metric, round_number)

# Create a federated model using the pytorch class, lambda optimizer function, and loss function
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
        'aggregator.settings.rounds_to_train': 300,
        # 'aggregator.settings.log_metric_callback': write_metric_x,
        # "aggregator.settings.write_logs": True,
    }
)

final_fl_model.save_native('final_income_model')

import sklearn

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

train = pd.read_csv("/Users/zhiruzhu/Desktop/data_station/fl_test/income/train.csv")
# train = pd.read_csv("/home/zhiru_uchicago_edu/federated_learning_test/income/train.csv")

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

# X_test = one_hot(X_test, 2)
# y_test = one_hot(y_test, 2)
# print(y_train)
# print(y_test)

X_train=torch.from_numpy(X_train.astype(np.float32))
X_test=torch.from_numpy(X_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(int))
y_test=torch.from_numpy(y_test.astype(int))

# batch_size = 32
# num_classes = 2

# fl_data = FederatedDataSet(X_train, y_train, X_test, y_test, batch_size=batch_size)  # , num_classes=num_classes)

input_dim = int(X_train.shape[1])
# print(input_dim)
output_dim = 1


class LogisticRegression(torch.nn.Module):

    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)  # .float()
        # self.layer1 = torch.nn.Linear(input_dim, 20)
        # self.layer2 = torch.nn.Linear(20, output_dim)

    def forward(self, x):
        # outputs = self.layer1(x)
        # outputs = torch.sigmoid(self.layer2(outputs))
        outputs = torch.sigmoid(self.linear(x))  # .float()
        return outputs


learning_rate = 0.01
optimizer = lambda x: torch.optim.SGD(x, lr=learning_rate)

def cross_entropy(output, target):
    """Binary cross-entropy metric
    """
    # criterion = nn.CrossEntropyLoss()
    # loss = criterion(output, target)
    # return loss
    output = torch.squeeze(output)
    #     output = output.float()
    return F.binary_cross_entropy(input=output.float(), target=target.float())

model = LogisticRegression()
optimizer = optimizer(model.parameters())
# optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

number_of_epochs=10
losses = []
losses_test = []

for epoch in range(number_of_epochs):
    y_prediction=model(X_train)
    loss=cross_entropy(y_prediction,y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # if (epoch+1)%10 == 0:
    #     print('epoch:', epoch+1,',loss=',loss.item())

    with torch.no_grad():

        # Calculating the loss and accuracy for the test dataset
        correct_test = 0
        total_test = 0
        outputs_test = torch.squeeze(model(X_test))
        loss_test = cross_entropy(outputs_test, y_test)

        predicted_test = outputs_test.round().detach().numpy()
        total_test += y_test.size(0)
        correct_test += np.sum(predicted_test == y_test.detach().numpy())
        accuracy_test = 100 * correct_test / total_test
        losses_test.append(loss_test.item())

        # Calculating the loss and accuracy for the train dataset
        total = 0
        correct = 0
        total += y_train.size(0)
        correct += np.sum(torch.squeeze(y_prediction).round().detach().numpy() == y_train.detach().numpy())
        accuracy = 100 * correct / total
        losses.append(loss.item())
        # Iterations.append(iter)

        print(f"Iteration: {epoch}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}")
        print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")

        # y_pred = model(X_test)
        # y_pred_class = y_pred.round()
        # print(y_pred_class.eq(y_test).sum())
        # print(len(y_test))
        # # print(y_test.shape[1])
        #
        # accuracy = (y_pred_class.eq(y_test).sum()) / float(y_test.shape[0])
        # print("acc: ", accuracy.item())

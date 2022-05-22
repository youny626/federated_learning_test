import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import openfl.native as fx
from openfl.federated import FederatedModel,FederatedDataSet

log_file = "/home/zhiru_uchicago_edu/.local/workspace/cifar.log"
if os.path.exists(log_file):
    os.remove(log_file)

#Setup default workspace, logging, etc.
fx.init('torch_cnn_mnist', log_level='METRIC', log_file="./logs/cifar.log")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

train_data =  train_set.data
train_data = train_data.transpose(0, 3, 1, 2).astype(np.float32)

train_labels = np.array(train_set.targets)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

def one_hot(labels, classes):
    return np.eye(classes)[labels]

test_data = test_set.data
test_data = test_data.transpose(0, 3, 1, 2).astype(np.float32)

test_labels = one_hot(np.array(test_set.targets), len(test_set.classes))

num_classes = len(train_set.classes)

batch_size = 32
fl_data = FederatedDataSet(train_data, train_labels, test_data, test_labels, batch_size=batch_size,
                           num_classes=num_classes)


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


optimizer = lambda x: optim.SGD(x, lr=0.001, momentum=0.9)

def cross_entropy(output, target):
    """Binary cross-entropy metric
    """
    # criterion = nn.CrossEntropyLoss()
    # loss = criterion(output, target)
    # return loss
    return F.cross_entropy(input=output, target=target)

import time


def write_metric(node_name, task_name, metric_name, metric, round_number):
    with open("./logs/cifar_openfl.log", "a") as log:
        log.write("{}/{}/{}: {}, {}, {}\n".format(node_name, task_name, metric_name, metric, round_number, time.time()))

#Create a federated model using the pytorch class, lambda optimizer function, and loss function
fl_model = FederatedModel(build_model=Net,optimizer=optimizer,loss_fn=cross_entropy,data_loader=fl_data)

num_collaborators = 8
collaborator_models = fl_model.setup(num_collaborators=num_collaborators)
collaborators = {}
for i in range(num_collaborators):
    collaborators[i] = collaborator_models[i]

print(f'Original training data size: {len(train_data)}')
print(f'Original validation data size: {len(test_data)}\n')

for i, model in enumerate(collaborator_models):
    print(f'Collaborator {i}\'s training data size: {len(model.data_loader.X_train)}')
    print(f'Collaborator {i}\'s validation data size: {len(model.data_loader.X_valid)}\n')

# Run experiment, return trained FederatedModel

final_fl_model = fx.run_experiment(collaborators, override_config={
    'aggregator.settings.rounds_to_train': 300,
    # 'aggregator.settings.log_metric_callback': write_metric,
})

#Save final model
final_fl_model.save_native('final_cifar_model')
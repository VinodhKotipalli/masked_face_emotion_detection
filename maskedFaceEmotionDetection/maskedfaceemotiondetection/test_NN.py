import matplotlib.pyplot as plt
import numpy as np
import random
import os

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

input_dim = 48 * 48
layer_dim = 1024
output_dim = 7


class Net1(nn.Module):
    # two layer NN using relu activation
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(input_dim, layer_dim)
        self.output = nn.Linear(layer_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.output(x)
        return x


class Net2(nn.Module):
    # three layer NN using relu activation
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(input_dim, layer_dim)
        self.fc2 = nn.Linear(layer_dim, layer_dim)
        self.output = nn.Linear(layer_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.output(x)
        return x


class Net3(nn.Module):
    # three layer NN using sigmoid activation
    def __init__(self):
        super(Net3, self).__init__()
        self.fc1 = nn.Linear(input_dim, layer_dim)
        self.fc2 = nn.Linear(layer_dim, layer_dim)
        self.output = nn.Linear(layer_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.output(x)
        return x


def calc_accuracy(labels, output):
    correct = 0.
    output = output.detach().numpy()
    for y in range(len(labels)):
        pred = np.argmax(output[y])
        if labels[y] == pred:  # true pos and true neg
            correct += 1
    return correct / len(labels)  # accuracy = tp + tn / total


def preprocess(data):
    # print("Scaling raw data..")
    # scale data by 255 except for last element (class)
    data[:, :-1] /= 255
    # return all data
    # print("Data scaled.")
    return data


if __name__ == '__main__':
    # find directory to csv files of images
    __file__ = os.getcwd()
    proj_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_dir = os.path.join(proj_file, "datasets")
    testData = []
    trainData = []

    # add each csv for masked and unmasked training and test data
    for filename in os.listdir(csv_dir):
        if filename.endswith(".csv"):
            f = os.path.join(csv_dir, filename)
            if "test" in filename:
                testData.append(f)
            else:
                trainData.append(f)
        else:
            continue

    # iterate through masked and unmasked
    for i in range(len(trainData)):
        # load test and training data from csv
        print("Loading training and test data ...")
        testset = np.loadtxt(testData[i], dtype=str, delimiter=',')
        trainset = np.loadtxt(trainData[i], dtype=str, delimiter=',')

        # convert data from str to floating point
        testset = np.float32(testset)
        trainset = np.float32(trainset)

        # randomly shuffle the training and test data
        np.random.shuffle(testset)
        np.random.shuffle(trainset)

        print("Scaling data ...")
        # preprocess the training and test data to scale values
        trainset = preprocess(trainset)
        testset = preprocess(testset)

        learning_rate = 0.01
        # learning_rate = 0.001
        batchsize = 30
        momentum_term = 0.5
        # momentum_term = 0.0

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                                  shuffle=True, num_workers=0)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                                 shuffle=False, num_workers=0)

        net1 = Net1()
        net2 = Net2()
        net3 = Net3()

        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.CrossEntropyLoss()
        criterion3 = nn.CrossEntropyLoss()

        optimizer1 = optim.SGD(net1.parameters(), lr=learning_rate,
                               momentum=momentum_term)
        optimizer2 = optim.SGD(net2.parameters(), lr=learning_rate,
                               momentum=momentum_term)
        optimizer3 = optim.SGD(net3.parameters(), lr=learning_rate,
                               momentum=momentum_term)

        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0

        epochs = 100

        accuracies1 = []
        accuracies2 = []
        accuracies3 = []

        print("Training Beginning for %d epochs" % epochs)
        print("Batch size ", batchsize)
        print("Learning rate ", learning_rate)

        for epoch in range(epochs):  # loop over the dataset multiple times
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data[:, :-1]
                labels = data[:, -1]
                labels = labels.long()

                # zero the parameter gradients
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()

                # forward + backward + optimize
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)
                outputs3 = net3(inputs)
                loss1 = criterion1(outputs1, labels)
                loss1.backward(retain_graph=True)
                loss2 = criterion2(outputs2, labels)
                loss2.backward(retain_graph=True)
                loss3 = criterion3(outputs3, labels)
                loss3.backward(retain_graph=True)
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()

                running_loss1 += loss1.item()
                running_loss2 += loss2.item()
                running_loss3 += loss3.item()

                if i % 100 == 99:  # every 1000 mini-batches...

                    # ...log the running loss
                    print('[%d, %5d] loss 1: %.20f' %
                          (epoch + 1, i + 1, running_loss1 / 100))
                    running_loss1 = 0.0
                    print('[%d, %5d] loss 2: %.20f' %
                          (epoch + 1, i + 1, running_loss2 / 100))
                    running_loss2 = 0.0
                    print('[%d, %5d] loss 3: %.20f' %
                          (epoch + 1, i + 1, running_loss3 / 100))
                    running_loss3 = 0.0

            print('Finished Training epoch:', epoch)

            for data in testloader:
                images = data[:, :-1]
                labels = data[:, -1]

                output1 = net1(images)
                output2 = net2(images)
                output3 = net3(images)

                accuracy1 = calc_accuracy(labels, output1)
                accuracy2 = calc_accuracy(labels, output2)
                accuracy3 = calc_accuracy(labels, output3)

                print('Accuracy 1', accuracy1)
                print('Accuracy 2', accuracy2)
                print('Accuracy 3', accuracy3)

                accuracies1.append(accuracy1)
                accuracies2.append(accuracy2)
                accuracies3.append(accuracy3)

            print("_______________________________________________________________")
            print()

        plt.plot(accuracies1, label="Net1")
        plt.plot(accuracies2, label="Net2")
        plt.plot(accuracies3, label="Net3")
        plt.show()
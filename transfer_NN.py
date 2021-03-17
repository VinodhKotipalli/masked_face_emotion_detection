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


class Net(nn.Module):
    # three layer NN using relu activation
    def __init__(self):
        super(Net, self).__init__()
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
    proj_file = os.path.abspath(__file__)
    csv_dir = os.path.join(proj_file, "datasets")
    testData = []
    trainData = []
    testData_mask = []
    trainData_mask = []

    # add each csv for masked and unmasked training and test data
    for filename in os.listdir(csv_dir):
        if filename.endswith(".csv"):
            f = os.path.join(csv_dir, filename)
            if "test" in filename:
                if "Without" in filename:
                    testData.append(f)
                else:
                    testData_mask.append(f)
            else:
                if "Without" in filename:
                    trainData.append(f)
                else:
                    trainData_mask.append(f)
        else:
            continue

    # load test and training data from csv
    print("Loading training and test data ...")
    testset = np.loadtxt(testData[0], dtype=str, delimiter=',')
    trainset = np.loadtxt(trainData[0], dtype=str, delimiter=',')

    testset_mask = np.loadtxt(testData_mask[0], dtype=str, delimiter=',')
    trainset_mask = np.loadtxt(trainData_mask[0], dtype=str, delimiter=',')

    # convert data from str to floating point
    testset = np.float32(testset)
    trainset = np.float32(trainset)
    testset_mask = np.float32(testset_mask)
    trainset_mask = np.float32(trainset_mask)

    # randomly shuffle the training and test data
    np.random.shuffle(testset)
    np.random.shuffle(trainset)
    np.random.shuffle(testset_mask)
    np.random.shuffle(trainset_mask)

    print("Scaling data ...")
    # preprocess the training and test data to scale values
    trainset = preprocess(trainset)
    testset = preprocess(testset)
    trainset_mask = preprocess(trainset_mask)
    testset_mask = preprocess(testset_mask)

    learning_rate = 0.01
    batchsize = 30
    momentum_term = 0.5

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                              shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                             shuffle=False, num_workers=0)
    trainloader_mask = torch.utils.data.DataLoader(trainset_mask, batch_size=batchsize,
                                                   shuffle=True, num_workers=0)
    testloader_mask = torch.utils.data.DataLoader(testset_mask, batch_size=len(testset_mask),
                                                  shuffle=False, num_workers=0)

    net = Net()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                          momentum=momentum_term)

    running_loss = 0.0

    epochs = 100

    accuracies = []
    max_acc = 0.0

    print("Initial Training Beginning for %d epochs" % epochs)
    print("Batch size ", batchsize)
    print("Learning rate ", learning_rate)

    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[:, :-1]
            labels = data[:, -1]
            labels = labels.long()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:  # every 1000 mini-batches...

                # ...log the running loss
                print('[%d, %5d] loss : %.20f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        print('Finished Training epoch:', epoch)

        for data in testloader:
            images = data[:, :-1]
            labels = data[:, -1]

            output = net(images)

            accuracy = calc_accuracy(labels, output)

            print('Accuracy', accuracy)

            accuracies.append(accuracy)
            if accuracy > max_acc:
                max_acc = accuracy
            print("_______________________________________________________________")
            print()

    plt.plot(accuracies, label="Initial Training")
    plt.show()

    print("Transfer Learning Beginning for %d epochs" % epochs)
    print("Batch size ", batchsize)
    print("Learning rate ", learning_rate)
    running_loss = 0.0
    accuracies_mask = []
    max_acc_mask = 0.0

    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader_mask, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[:, :-1]
            labels = data[:, -1]
            labels = labels.long()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:  # every 1000 mini-batches...

                # ...log the running loss
                print('[%d, %5d] loss : %.20f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        print('Finished Training epoch:', epoch)

        for data in testloader_mask:
            images = data[:, :-1]
            labels = data[:, -1]

            output = net(images)

            accuracy = calc_accuracy(labels, output)

            print('Accuracy', accuracy)

            accuracies_mask.append(accuracy)
            if accuracy > max_acc_mask:
                max_acc_mask = accuracy
            print("_______________________________________________________________")
            print()

    plt.plot(accuracies_mask, label="Transfer Learning")
    plt.show()
    print("Max accuracy for maskless is ", max_acc)
    print("Max accuracy for masks is ", max_acc_mask)

    print("Masked Training Beginning for %d epochs" % epochs)
    print("Batch size ", batchsize)
    print("Learning rate ", learning_rate)
    net = Net()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                          momentum=momentum_term)
    running_loss = 0.0
    accuracies_maskOnly = []
    max_acc_maskOnly = 0.0

    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader_mask, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[:, :-1]
            labels = data[:, -1]
            labels = labels.long()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:  # every 1000 mini-batches...

                # ...log the running loss
                print('[%d, %5d] loss : %.20f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        print('Finished Training epoch:', epoch)

        for data in testloader_mask:
            images = data[:, :-1]
            labels = data[:, -1]

            output = net(images)

            accuracy = calc_accuracy(labels, output)

            print('Accuracy', accuracy)

            accuracies_maskOnly.append(accuracy)
            if accuracy > max_acc_maskOnly:
                max_acc_maskOnly = accuracy

            print("_______________________________________________________________")
            print()

    plt.plot(accuracies_maskOnly, label="Mask Training")
    plt.show()
    print("Max accuracy for masks only is ", max_acc_maskOnly)
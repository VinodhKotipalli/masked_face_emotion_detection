#CS 541 Final Project
# Masked Face Emotion Detection - Neural Network
import numpy as np
import matplotlib.pyplot as plt
import os
import random


class Emotion:
    ANGRY       = 0
    DISGUST     = 1
    FEAR        = 2
    HAPPY       = 3
    NEUTRAL     = 4
    SAD         = 5
    SURPRISE    = 6


def preprocess(data):
    # print("Scaling raw data..")
    # scale data by 255 except for last element (class)
    data[:,:-1] /= 255
    # return all data
    # print("Data scaled.")
    return data


def train(traindata, weights, weights_hidden, delta, delta_hidden, learning_rate, momentum):
    # print("Beginning training..")
    # iterate through each training image
    for img in traindata:
        x = img[:-1]
        y = img[-1]

        # create array for actual target values for NN
        actual = np.array([0.1 for i in range(10)])
        # set actual value of data point to 1 in actual array to represent target value
        actual[int(y)] = 0.9

        pred, h = guess(x, weights, weights_hidden)

        # compare guess to actual values
        if not np.array_equal(pred,actual):
            # update weights if guess was incorrect
            weights, weights_hidden, delta, delta_hidden = optimize(weights, weights_hidden, delta, delta_hidden, x, h, pred, actual, learning_rate, momentum)

    # print("Training completed.")
    return weights, weights_hidden


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def optimize(weights, weights_hidden, delta, delta_hidden, x, h, guess, actual, learning_rate, momentum):
    # print("Optimizing weights..")

    output_error = np.array([0. for i in range(7)])
    hidden_error = np.array([0. for i in range(hidden_units + 1)])
    hidden_units_arr = np.concatenate(([1],h), axis=None)
    input_units_arr = np.concatenate(([1],x), axis=None)

    # calculate error with respect to each output
    for i in range(len(output_error)):
        output_error[i] = guess[i] * (1 - guess[i]) * (actual[i] - guess[i])

    # calculate error with respect to each hidden unit
    for i in range(len(hidden_error)):
        sum_units = 0

        for j in range(len(output_error)):
            sum_units += weights_hidden[j][i] * output_error[j]

        hidden_error[i] = hidden_units_arr[i] * (1 - hidden_units_arr[i]) * sum_units

    # update hidden weights
    for i in range(len(weights_hidden)):
        for j in range(len(weights_hidden[i])):
            # calculate delta for weights and save for next iteration
            temp = (learning_rate * output_error[i] * hidden_units_arr[j]) + (momentum * delta_hidden[i][j])
            delta_hidden[i][j] = temp
            # new weight = weight + delta
            weights_hidden[i][j] = weights_hidden[i][j] + temp

    # update initial weights
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            temp = (learning_rate * hidden_error[i] * input_units_arr[j]) + (momentum * delta[i][j])
            delta[i][j] = temp
            weights[i][j] = weights[i][j] + temp

    return weights, weights_hidden, delta, delta_hidden


def guess(data_x, weights, weights_hidden):
    # create array for guesses
    y = np.array([0. for i in range(7)])

    # create array for hidden input values
    hidden_x = np.array([0. for x in range(hidden_units)])

    # iterate through all hidden input units
    for j in range(len(hidden_x)):
        # calculate the value of the hidden input unit
        value = np.dot(weights[j], np.concatenate((1, data_x), axis=None))
        hidden_x[j] = sigmoid(value)

    # iterate through all 7 possible outputs
    for j in range(len(y)):
        value = np.dot(weights_hidden[j], np.concatenate((1, hidden_x), axis=None))
        y[j] = sigmoid(value)

    # NN prediction is the guess with the highest prediction value
    y_max = np.argmax(y)

    # iterate through all guesses
    for j in range(len(y)):
        # convert number value of guess to 0.9 or 0.1 depending on what is predicted by the perceptron
        if j == y_max:
            y[j] = 0.9
        else:
            y[j] = 0.1

    return y, hidden_x


def test(test_data, weights, weights_hidden):
    # print("Testing algorithm on test data..")

    # accuracy = tp + tn / tp + tn + fp + fn
    # numerator is all the correct values (tps and tns) where actual[i] == pred[i]
    # denominator is total guesses
    num = 0.
    den = 0.

    confusion = np.array([[0 for i in range(7)] for j in range(7)])
    for img in test_data:
        x = img[:-1]
        y = img[-1]

        actual = [0.1 for i in range(7)]
        actual[int(y)] = 0.9

        pred, _ = guess(x, weights, weights_hidden)
        predicted = np.argmax(pred)

        # print("Actual: " + str(y) + "\t Predicted: " + str(predicted))
        confusion[int(y)][predicted] += 1

        for j in range(len(actual)):
            if actual[j] == pred[j]:
                num += 1
                den += 1
            else:
                den += 1

    accuracy = num/den

    # print("Testing completed.")
    # print("Testing accuracy: ", accuracy)
    return accuracy, confusion


if __name__ == '__main__':
    # find directory to csv files of images
    proj_file = os.path.dirname(os.path.abspath(__file__))
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
        test_data = np.loadtxt(testData[i], dtype=str, delimiter=',')
        training_data = np.loadtxt(trainData[i], dtype=str, delimiter=',')

        # convert data from str to floating point
        test_data = np.float32(test_data)
        training_data = np.float32(training_data)

        # randomly shuffle the training and test data
        test_data = np.random.shuffle(test_data)
        training_data = np.random.shuffle(training_data)

        print("Scaling data ...")
        # preprocess the training and test data to scale values
        training_data = preprocess(training_data)
        test_data = preprocess(test_data)

        hidden_units = 100
        learning_rate = 0.1
        momentum = 0.9
        epochs = 50

        # nx2304 matrix of weights and corresponding matrix to track delta w for back propagation
        # each matrix row corresponds with the n hidden inputs that need to be calculated
        # each column corresponds with a weight for the input values 0-2304 plus one bias unit
        # weights initialized to a random value between -0.5 and 0.5
        weights = np.array([[random.uniform(-0.05, 0.05) for i in range(2304 + 1)] for j in range(hidden_units)])
        delta = np.array([[0. for i in range(2304 + 1)] for j in range(hidden_units)])

        # 7xn matrix of hidden weights and corresponding matrix to track delta w for back propagation
        # each row corresponds to an output 0-6
        # each column corresponds to a weight for the n hidden input values plus one bias unit
        # weights initialized to a random value between -0.5 and 0.5
        weights_hidden = np.array([[random.uniform(-0.05, 0.05) for i in range(hidden_units + 1)] for j in range(7)])
        delta_hidden = np.array([[0. for i in range(hidden_units + 1)] for j in range(7)])

        # array for plotting accuracy after each epoch
        accuracies = np.array([0. for i in range(epochs)])
        confusion_matrix = np.array([[0 for i in range(7)] for j in range(7)])

        # run epochs of training
        print("Starting training ...")
        for epoch in range(epochs):
            print("Running training epoch ", epoch)
            # train NN on training data for epochs
            weights, weights_hidden = train(training_data, weights, weights_hidden, delta, delta_hidden, learning_rate, momentum)

            # test NN on test data using weights from epoch
            print("Running testing epoch ", epoch)
            accuracy, confusion = test(test_data, weights, weights_hidden)

            # save accuracy from test data
            print("Epoch: " + str(epoch) + "\t Accuracy: " + str(accuracy))
            accuracies[epoch] = accuracy

            # save confusion matrix on last epoch
            if epoch == 49:
                confusion_matrix = confusion

        # plot accuracies vs epochs and save plot and confusion matrix
        plt.plot(accuracies)
        plt.savefig('NeuralNetwork' + str(i) + '.png')
        np.savetxt('NeuralNetwork' + str(i) + '.txt', confusion_matrix)
        plt.show()
        print(confusion_matrix)

        # look through a couple examples
        #for img in training_data:
        #    x = img[:-1]
        #    x *= 255
        #    y = img[-1]
        #    pred, _ = guess(x,weights,weights_hidden)
        #    prediction = np.argmax(pred)
        #    image = x.reshape((48, 48));
        #    plt.imshow(image, cmap="Greys");
        #    plt.title("Predicted: " + Emotion[prediction] + "\t Actual: " + Emotion[y])
        #    plt.show();

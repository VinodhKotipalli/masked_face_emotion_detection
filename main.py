#CS 541 Final Project
# Masked Face Emotion Detection - Neural Network
import numpy as np
import matplotlib.pyplot as plt


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def preprocess(data_x):
    print("Scaling raw data..")
    # scale data by 255
    data = data_x / 255
    # return all data
    print("Data scaled.")
    return data


def initialize(weights, weights_hidden):
    print("Initializing weights..")
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            # instantiate each weight to a random value -0.05 to 0.05
            temp = random.uniform(-0.05, 0.05)
            weights[i][j] = temp
    for i in range(len(weights_hidden)):
        for j in range(len(weights_hidden[i])):
            # instantiate each weight to a random value -0.05 to 0.05
            temp = random.uniform(-0.05, 0.05)
            weights_hidden[i][j] = temp
    # return with both weights matrices updated to initial values between -0.05 to 0.05
    print("Weights initialized.")
    #print(weights)
    #print(weights_hidden)
    return weights, weights_hidden


def train(traindata_x, traindata_y, weights, weights_hidden, delta, delta_hidden, learning_rate, momentum):
    print("Beginning training..")
    # iterate through each training image
    for i in range(len(traindata_x)):
        #print("Training on image ", i)
        #print("Actual value is ", training_y[i])
        #img = training_x[i].reshape((28, 28));
        #plt.imshow(img, cmap="Greys");
        #plt.show();

        # create array for actual target values for NN
        actual = np.array([0.1 for x in range(10)])
        # set actual value of data point to 1 in actual array to represent target value
        actual[np.int32(traindata_y[i])] = 0.9

        y, h = guess(traindata_x[i], weights, weights_hidden)
        #print("Predicted value is ", np.argmax(y))
        #print("actual: ", actual)
        #print("y: ", y)
        #print("h: ", h)
        # compare guess to actual values
        if not np.array_equal(y,actual):
            #print("Predicted value is incorrect, beginning optimization..")
            # update weights if guess was incorrect
            weights, weights_hidden, delta, delta_hidden = optimize(weights, weights_hidden, delta, delta_hidden, traindata_x[i], h, y, actual, learning_rate, momentum)
            #print("Hidden weights", weights_hidden)

    print("Training completed.")
    return weights, weights_hidden


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def optimize(weights, weights_hidden, delta, delta_hidden, x, h, guess, actual, learning_rate, momentum):
    print("Optimizing weights..")

    output_error = np.array([0. for i in range(10)])
    hidden_error = np.array([0. for i in range(hidden_units + 1)])
    hidden_units_arr = np.concatenate(([1],h), axis=None)
    input_units_arr = np.concatenate(([1],x), axis=None)

    #print("Computing output error..")
    for i in range(len(output_error)):
        output_error[i] = guess[i] * (1 - guess[i]) * (actual[i] - guess[i])
        #print(output_error)

    #print("Computing hidden error..")
    for i in range(len(hidden_error)):
        sum_units = 0

        for j in range(len(output_error)):
            sum_units += weights_hidden[j][i] * output_error[j]

        hidden_error[i] = hidden_units_arr[i] * (1 - hidden_units_arr[i]) * sum_units
        #print(hidden_error)

    #print("Updating hidden weights..")
    for i in range(len(weights_hidden)):
        for j in range(len(weights_hidden[i])):
            # calculate delta for weights and save for next iteration
            temp = (learning_rate * output_error[i] * hidden_units_arr[j]) + (momentum * delta_hidden[i][j])
            delta_hidden[i][j] = temp
            # new weight = weight + delta
            weights_hidden[i][j] = weights_hidden[i][j] + temp
    #print(weights_hidden)

    #print("Updating initial weights..")
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            temp = (learning_rate * hidden_error[i] * input_units_arr[j]) + (momentum * delta[i][j])
            delta[i][j] = temp
            weights[i][j] = weights[i][j] + temp
    #print(weights)

    #print("Weights optimized.")
    return weights, weights_hidden, delta, delta_hidden


def guess(data_x, weights, weights_hidden):
    #print("Algorithm making prediction..")
    # create array for guesses
    y = np.array([0. for i in range(10)])

    # create array for hidden input values
    hidden_x = np.array([0. for x in range(hidden_units)])
    # set first value of of hidden inputs to +1 bias value
    #hidden_x[0] = 1

    #print("Calculating hidden input units..")
    # iterate through all hidden input units
    for j in range(len(hidden_x)):
        # calculate the value of the hidden input unit
        value = np.dot(weights[j], np.concatenate((1, data_x), axis=None))
        #print(value)
        hidden_x[j] = sigmoid(value)

    #print("Hidden units: ", hidden_x)

    #print("Calculating possible outputs..")
    # iterate through all 10 possible outputs
    for j in range(len(y)):
        value = np.dot(weights_hidden[j], np.concatenate((1, hidden_x), axis=None))
        #print(value)
        y[j] = sigmoid(value)

    # NN prediction is the guess with the highest prediction value
    y_max = np.argmax(y)
    #print(y)
    #print(y_max)

    # iterate through all guesses
    for j in range(len(y)):
        # convert number value of guess to 0.9 or 0.1 depending on what is predicted by the perceptron
        if j == y_max:
            y[j] = 0.9
        else:
            y[j] = 0.1

    #print("Prediction complete.")
    return y, hidden_x


def test(test_x, test_y, weights, weights_hidden):
    print("Testing algorithm on test data..")
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    confusion = np.array([[0 for i in range(10)] for j in range(10)])
    for i in range(len(test_x)):
        #print("Training on image ", i)
        target = np.array([0.1 for i in range(10)])
        target[np.int32(test_y[i])] = 0.9

        y, hidden_x = guess(test_x[i], weights, weights_hidden)

        actual = np.int32(test_y[i])
        predicted = np.argmax(y)
        #print("Actual value is ", actual)
        #print("Predicted value is ", predicted)

        confusion[actual][predicted] = confusion[actual][predicted] + 1

        for j in range(len(target)):
            if target[j] == 0.9:
                if y[j] == 0.9:
                    tp += 1
                else:
                    fn += 1
            else:
                if y[j] == 0.9:
                    fp += 1
                else:
                    tn += 1

    accuracy = (tp + tn)/(tp + tn + fp + fn)
    print("Testing completed.")
    print("Testing accuracy: ", accuracy)
    return accuracy, confusion


if __name__ == '__main__':

    test_data = np.loadtxt('mnist_test.csv', dtype=str, delimiter=',')
    training_data = np.loadtxt('mnist_train.csv', dtype=str, delimiter=',')

    training_data = np.float32(training_data[1:])
    training_x = training_data[:, 1:]
    training_y = training_data[:, 0]

    test_data = np.float32(test_data[1:])
    test_x = test_data[:, 1:]
    test_y = test_data[:, 0]

    # preprocess the training data to scale values
    training_x = preprocess(training_x)

    #Experiment 1, change hidden units
    hidden_units_arr = [50,100]
    for hidden_units in hidden_units_arr:
        learning_rate = 0.1
        momentum = 0.9

        # nx785 matrix of weights and corresponding matrix to track delta w for back propagation
        # each matrix row corresponds with the n hidden inputs that need to be calculated
        # each column corresponds with a weight for the input values 0-784
        weights = np.array([[0. for i in range(785)] for j in range(hidden_units)])
        delta = np.array([[0. for i in range(785)] for j in range(hidden_units)])
        #print(weights.shape)

        # 10xn matrix of hidden weights and corresponding matrix to track delta w for back propagation
        # each row corresponds to an output 0-9
        # each column corresponds to a weight for the n hidden input values plus one bias unit
        weights_hidden = np.array([[0. for i in range(hidden_units + 1)] for j in range(10)])
        delta_hidden = np.array([[0. for i in range(hidden_units + 1)] for j in range(10)])
        #print(weights_hidden.shape)

        # initialize all weights to random values
        weights, weights_hidden = initialize(weights, weights_hidden)

        # array for plotting accuracy after each epoch
        accuracies = np.array([0. for i in range(50)])
        confusion_matrix = np.array([[0 for i in range(10)] for j in range(10)])

        # run 50 epochs of training
        for i in range(50):
            print("Running training epoch ", i)
            # train NN on training data for 50 epochs
            weights, weights_hidden = train(training_x, training_y, weights, weights_hidden, delta, delta_hidden, learning_rate, momentum)
            # test NN on test data using weights from epoch
            accuracy, confusion = test(test_x, test_y, weights, weights_hidden)
            # save accuracy from test data
            accuracies[i] = accuracy
            if i == 49:
                confusion_matrix = confusion

        plt.plot(accuracies)
        plt.savefig('Exp1_hiddenunits' + str(hidden_units) + '.png')
        #plt.show()
        #print(confusion_matrix)
        np.savetxt('Exp1_hiddenunits' + str(hidden_units) + '.txt', confusion_matrix)
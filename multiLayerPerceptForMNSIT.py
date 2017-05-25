import numpy as np
from mnist import MNIST
import sklearn

mndata = MNIST('data')

imagesTrain, labelsTrain = mndata.load_training()
imagesTest , labelsTest  = mndata.load_testing()

# helps in checking the data properties:
# print(type(imagesTrain))
# print(len(imagesTrain))
# print(imagesTrain[0])
# print(type(imagesTrain[0]))
# print(len(imagesTrain[0]))
# print(imagesTrain[0][20])
# print(labelsTrain)

labelTrainOneHot = np.zeros((len(labelsTrain),10))

for images in range(len(labelsTrain)):
    labelTrainOneHot[images, int(labelsTrain[images])] = 1

xInput = np.array(imagesTrain)
yInput = labelTrainOneHot

xInputTest = np.array(imagesTest)

def predict(x, theta):
    return 1/(1 + np.exp(- np.matmul(x, theta)))


def costFunction(y, yPredicted):
    yPredicted[np.where(yPredicted < 1e-300)] = 1e-300
    yPredicted[np.where((1 - yPredicted) < 1e-300)] = 1e-300
    return np.sum(- np.multiply(y, np.log(yPredicted)) - np.multiply((1-y), np.log(yPredicted)))


def gradFunction(x, y, theta):
    yPredicted = predict(x, theta)
    return np.matmul(x.T, np.subtract(yPredicted, y)) / y.shape[0]


def getDeltaForTheta(deltaForOpLayer, input):
    return np.matmul(input.T, deltaForOpLayer) / input.shape[0]


def getDeltaForOpLayer(yPredicted, yInput):
    return -np.multiply(np.subtract(yInput, yPredicted), np.multiply(yPredicted, 1-yPredicted))


def getDeltaForHidLayer(deltaOfNextLayer, theta, opOfHidLayer):
    return np.multiply(np.matmul(deltaOfNextLayer, theta.T), np.multiply(opOfHidLayer, 1-opOfHidLayer))

alpha = 0.1
n_epocs = 10000
theta1 = np.random.random((xInput.shape[1]+1, 10)) - 0.5
xInput = (np.insert(xInput, 0, 1, axis=1) ) / 256

mean = np.array([xInput.mean(axis=0)])
print(mean.shape)
mean = np.matmul(np.ones((60000, 1)), mean)
xInput = np.subtract(xInput, mean)

y1 = np.matmul(xInput, theta1)

acc = list()
preCost = 0
for iterations in range(n_epocs):
    temp = iterations%1000
    y1 = predict(xInput, theta1)
    deltaOpLayer = getDeltaForOpLayer(y1, yInput)
    # print("y1", y1[0:20,:])
    # print("yInput", yInput[0:20,:])

    # print(deltaOpLayer[0:20,:])
    deltaTheta1 = getDeltaForTheta(deltaOpLayer, xInput)
    tempTheta1 = theta1 - alpha * deltaTheta1
    theta1 = tempTheta1
    # print("theta1", theta1[0:20,:])

    print("Done for iteration number : ", iterations)
    # get cost
    if(iterations%1000 == 0):
        y1 = predict(xInput, theta1)
        # print("y1", y1[0:20, :])
        deltaOpLayer = getDeltaForOpLayer(y1, yInput)
        # print("deltaOpLayer", deltaOpLayer[0:20, :])
        cost = costFunction(yInput, y1)
        print("Iteration, cost", iterations, cost, preCost-cost)
        print(yInput.argmax(axis=1).shape)
        temp = sum(yInput.argmax(axis=1) == y1.argmax(axis=1)) / 600
        print( temp )
        acc.append([iterations, temp])

        # debug = input()
        # if(debug == 'y'):
        #     break
        preCost = cost

for i in range(len(acc)):
    print(acc[i])

"""
Some stats :

Iteration, cost, change : 1000 634136.017759 -2144.25288967
Accuracy : 71.915
Iteration, cost, change : 2000 664346.465893 -30210.4481347
Accuracy : 74.2416666667

"""

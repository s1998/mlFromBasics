import numpy as np

# multi layer perceptron for XOR gate, treating it as
# binary classification problem and classifying output as 0 or 1
# reference for back-propagation : https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

# Creating input
xInput = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
yInput = np.array([[0], [1], [1], [0]])
# print(xInput)
# print(yInput)

# Adding the bais variable
x = np.insert(xInput, 0, 1, axis=1)
# print(x)


def predict(x, theta):
    return 1/(1 + np.exp(- np.matmul(x, theta)))


def costFunction(y, yPredicted):
    return np.sum(- np.multiply(y, np.log(yPredicted)) - np.multiply((1-y), np.log(1-yPredicted))) / y.shape[0]


def gradFunction(x, y, theta):
    yPredicted = predict(x, theta)
    return np.matmul(x.T, np.subtract(yPredicted, y)) / y.shape[0]


def getDeltaForTheta(deltaForOpLayer, input):
    return np.matmul(input.T, deltaForOpLayer)


def getDeltaForOpLayer(yPredicted, yInput):
    # print("yInput :           ", yInput.shape)
    # print("yPredicted shape : ", yPredicted.shape)
    return -np.multiply(np.subtract(yInput, yPredicted), np.multiply(yPredicted, 1-yPredicted))


def getDeltaForHidLayer(deltaOfNextLayer, theta, opOfHidLayer):
    return np.multiply(np.matmul(deltaOfNextLayer, theta.T), np.multiply(opOfHidLayer, 1-opOfHidLayer))

alpha = 0.1
n_epocs = 1000000

# Two layer network for xor, has 2 units in the hidden layer
# lets call output of hidden layer as y1
theta1 = np.random.random((xInput.shape[1]+1, 2)) + 2#+ np.ones((xInput.shape[1]+1, 2))*0.00001
print(theta1.shape)
y1 = np.matmul(x, theta1)
theta2 = np.random.random((y1.shape[1]+1, 1)) + 2 #+ np.ones((y1.shape[1]+1, 2))*0.00001

for iterations in range(n_epocs):
    # get out_h1 output of hidden layer h1
    y1 = predict(x, theta1) # (4*3)*(3*2) -> (4*2)
    # add bias to it
    y1 = np.insert(y1, 0, 1, axis=1) # (4*3)
    # get output of the second layer
    y2 = predict(y1, theta2) # (4*3)*()

    # get delta for the output layer
    deltaOpLayer = getDeltaForOpLayer(y2, yInput)
    # print("deltaOpLayer.shape : ", deltaOpLayer.shape)
    # print("deltaOpLayer", deltaOpLayer)

    # get delta for theta2
    deltaTheta2 = getDeltaForTheta(deltaOpLayer, y1)
    # print("deltaTheta2.shape : ", deltaTheta2.shape)
    # print("deltaTheta2", deltaTheta2)

    # get delta for hidden layer
    deltaH1Layer = getDeltaForHidLayer(deltaOpLayer, theta2, y1)
    # print("deltaH1Layer", deltaH1Layer)

    # remove the error of bias, beacuse theta1 is not
    # responsible for the bias in next layer
    deltaH1Layer = np.delete(deltaH1Layer, [0], axis=1)
    # print("deltaH1Layer.shape : ", deltaH1Layer.shape)
    # print("deltaH1Layer", deltaH1Layer)

    # get delta for theta1
    deltaTheta1 = getDeltaForTheta(deltaH1Layer, x)
    # print("deltaTheta1", deltaTheta1)
    # print("deltaTheta1.shape : ", deltaTheta1.shape)

    # update the parameters
    # print("theta1.shape", theta1.shape)
    # print("theta2.shape", theta2.shape)
    tempTheta2 = theta2 - alpha * deltaTheta2
    tempTheta1 = theta1 - alpha * deltaTheta1
    theta1 = tempTheta1
    theta2 = tempTheta2
    # get cost
    y1 = predict(x, theta1)
    y1 = np.insert(y1, 0, 1, axis=1)
    y2 = predict(y1, theta2)
    cost = costFunction(yInput, y2)
    if(iterations%10000 == 0):
        print("Iteration, cost", iterations, cost)
        print(yInput, y2)
        debig = input()

'''
Results for XOR network
xInput = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
yInput = np.array([[0], [1], [1], [0]])

Some results :
Iteration, cost, predictions : 0 4.07246596989
[[ 0.99960371]
 [ 0.99970961]
 [ 0.99970979]
 [ 0.99971547]]

Iteration, cost, predictions : 10000 0.522858400785
[[ 0.87404593]
 [ 0.35111276]
 [ 0.35052321]
 [ 0.33530097]]

Iteration, cost, predictions : 20000 0.486574047976
[[ 0.95495698]
 [ 0.33755838]
 [ 0.33556146]
 [ 0.33974108]]

Iteration, cost, predictions : 30000 0.0434984041827
[[ 0.95864794]
 [ 0.04114199]
 [ 0.04116308]
 [ 0.95340481]]

Iteration, cost, predictions : 40000 0.0279048283186
[[ 0.97157156]
 [ 0.02638406]
 [ 0.02640037]
 [ 0.97113919]]

'''


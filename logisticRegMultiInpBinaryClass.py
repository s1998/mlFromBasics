import numpy as np

xInput = np.random.random((1000, 6))
yInput = np.random.random_integers(low=0, high=1, size=(1000, 1))

theta = np.random.random((xInput.shape[1]+1, 1))

# yPredicted = hFunction(theta, x)
def hFunction(theta, x):
    return 1 / (1+np.exp(-np.matmul(x, theta)))

# print(predict([1], [ [1], [-1]])) shows : [ 0.73105858  0.26894142]
def predict(theta, x):
    return hFunction(theta, x)

# cost = y*log(hFucntion(theta, x)) + (1-y)*log(1-hFunction(theta, x))
def costFunction(y, yPredicted):
    # return 1 - y
    # return y * np.log(yPredicted)
    return np.sum(-(y * np.log(yPredicted) + (1 - y) * np.log(1 - yPredicted))) / y.shape[0]

# Checking cost function
# print(costFunction(np.array([1,1]), np.array([0.9,0.9])), np.log(0.9))

# adding the Bias Variable
x = np.ones((xInput.shape[0], xInput.shape[1]+1))
x[:, 1:] = xInput

def gradientOfTheta(x, y, theta):
    yPredicted = predict(theta, x)
    diff = yPredicted - y
    return np.matmul(x.T, diff)/y.shape[0]

# step size steeing
alpha = 0.0001

for iterations in range(1000000):
    tempTheta = theta - alpha * gradientOfTheta(x, yInput, theta)
    theta = tempTheta
    print("Iteration number, cost : ",iterations, costFunction(yInput, predict(theta, x)))
    # pass

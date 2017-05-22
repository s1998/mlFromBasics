import numpy as np

xInput = np.random.random((1000, 2))
yInput = np.zeros((xInput.shape[0], 1))
yInput[np.where(xInput[:,0] + xInput[:,1]>1.0)] = 1

theta = np.zeros((xInput.shape[1]+1, 1)) + 0.0001

# yPredicted = hFunction(theta, x)
def hFunction(theta, x):
    return 1 / (1+np.exp(-np.matmul(x, theta)))

# print(predict([1], [ [1], [-1]])) shows : [ 0.73105858  0.26894142]
def predict(theta, x):
    return hFunction(theta, x)

# cost = y*log(hFucntion(theta, x)) + (1-y)*log(1-hFunction(theta, x))
def costFunction(y, yPredicted):
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
alpha = 0.1

print(costFunction(yInput, predict(theta, x)))

for iterations in range(100000):
    tempTheta = theta - alpha * gradientOfTheta(x, yInput, theta)
    theta = tempTheta
    if(iterations%1000 == 0):
        print("Iteration number, cost : ",iterations, costFunction(yInput, predict(theta, x)))

print(costFunction(yInput, predict(theta, x)))

yPredicted = predict(theta, x)
for i in range(yInput.shape[0]):
    print(x[i],yInput[i], yPredicted[i])


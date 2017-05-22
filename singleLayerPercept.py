import numpy as np

# single layer perceptron for basic gates, treating them as
# continuous output instead of classification into 0 or 1

# Custom input
xInput = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
yInput = np.array([[0], [0], [0], [1]])
# print(xInput)
# print(yInput)

# Adding the bais variable
x = np.ones((xInput.shape[0], xInput.shape[1]+1))
x[:, 1:] = xInput
# print(x)

theta = np.zeros((x.shape[1], 1))

def predict(x, theta):
    return (1 / (1 + np.exp(- np.matmul(x, theta))))

def costFunction(y, yPredicted):
    return np.sum(- y*np.log(yPredicted) - (1-y)*np.log(1-yPredicted)) / y.shape[0]

def gradFunction(x, y, theta):
    yPredicted = predict(x, theta)
    return np.matmul(x.T, (yPredicted - y)) / y.shape[0]
    # pass

alpha = 0.001
n_epocs = 1000000
theta = np.random.random((x.shape[1], yInput.shape[1]))

x = np.ones((xInput.shape[0], xInput.shape[1]+1))
x[:,1:] = xInput
for iterations in range(n_epocs):
    tempTheta = theta - alpha * gradFunction(x, yInput, theta)
    theta = tempTheta
    print(iterations, costFunction(yInput, predict(x, theta)))

print(costFunction(yInput, predict(x, theta)) )
print(predict(x, theta))

'''
Results for OR network
xInput = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
yInput = np.array([[0], [1], [1], [1]])
Output for n_epoch = 1000000, alpha = 0.001
cost = 0.00928288
predict(x, theta) is : [[ 0.02048609] [ 0.99181996] [ 0.99181606] [ 0.99999858]]
'''

'''
Results for AND network
xInput = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
yInput = np.array([[0], [0], [0], [1]])
Output for n_epoch = 1000000, alpha = 0.001
cost = 0.017430789645
predict(x, theta) is : [[  1.24819001e-05] [  2.02565376e-02] [  2.02565298e-02] [  9.71628591e-01]]'''



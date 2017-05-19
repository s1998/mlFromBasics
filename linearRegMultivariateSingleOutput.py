import numpy as np

# The data is not normalized,
x = np.random.random((1000,5))
x = x.astype(float)
y = np.matrix(x[:,0]*1 + x[:,1]*2 + x[:,2]*3 + x[:,3]*4 + x[:,4]*5 + 6, dtype=float)
y = y.T

# Intialize the parameters
theta0 = np.zeros((x.shape[1],1), dtype=float)
theta1 = 0.00

# cost function
# yPredicted = np.matmul(x, theta0) + theta1;
# cost = ((yPredicted - y) ** 2) / (2 * y.shape[0])

#step -size:
alpha = 0.01

def predictFunction(x, theta0, theta1):
    return np.matmul(x, theta0) + theta1

def costFunction(y, yPredicted):
    return (np.power((y - yPredicted), 2) ).mean() / (2)

# Input is theta0, theta1, data input x, data input y
# i,j correspond to theta(i,j) the component wrt which we want to find the
# gradient of the cost function.
# In this code j = 1 always
def gradientFunctionIntution(x, theta0, theta1, y, i, j):
    return (x[:,i]*(predictFunction(x, theta0, theta1) - y)).mean()

def gradientFunction(x, diff):
    return np.matmul(x.T, diff) / diff.shape[0]

for iterations in range(10000):
    tempTheta0 = (theta0 - alpha * gradientFunction(x, predictFunction(x, theta0, theta1) - y))
    tempTheta1 = (theta1 - alpha * (predictFunction(x, theta0, theta1) - y).mean())
    theta0 = tempTheta0
    theta1 = tempTheta1

    print("Iteration number : ", iterations)
    print("Cost : ", costFunction(y, predictFunction(x, theta0, theta1)))
    print("Theta0 : ", theta0.T)
    print("Theta1 : ", theta1)


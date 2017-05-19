import numpy as np

# The data is not normalized,
x = np.random.random((10,1))
y = 20*x+3

# Intialize the parameters
theta0 = 0
theta1 = 0
y_predicted = theta0 * x + theta1

#step -size:
alpha = 0.01

# Cost function
cost = ((y_predicted - y)**2).mean() / 2

def predict(x,theta0, theta1):
    return x*theta0+theta1

def cost(y, y_predicted):
    return ((y_predicted - y)**2).mean() / 2

for iterations in range(100000):
    tempTheta0 = theta0 - alpha * (x * (predict(x, theta0, theta1) - y)).mean()
    tempTheta1 = theta1 - alpha * (predict(x, theta0, theta1) - y).mean()
    theta0 = tempTheta0
    theta1 = tempTheta1
    print("Iteration number : ", iterations, "Cost : ", cost(predict(x,theta0, theta1), y), "Theta0 : ", theta0, "Theta1 : ", theta1 )



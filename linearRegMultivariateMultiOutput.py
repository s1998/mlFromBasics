import numpy as np

# The data is not normalized,
x = np.random.random((100,5))
x = x.astype(float)
y = np.ones((x.shape[0], 3), dtype=float)
y[:,0] = np.array(x[:,0]*1 + x[:,1]*2 + x[:,2]*3 + x[:,3]*4 + x[:,4]*5 + 6, dtype=float).T
y[:,1] = np.array(x[:,0]*2 + x[:,1]*3 + x[:,2]*4 + x[:,3]*5 + x[:,4]*6 + 7, dtype=float).T
y[:,2] = np.array(x[:,0]*3 + x[:,1]*4 + x[:,2]*5 + x[:,3]*6 + x[:,4]*7 + 8, dtype=float).T

theta0 = np.zeros((5,y.shape[1]), dtype=float)
theta1 = np.zeros((1,y.shape[1]), dtype=float)
print("Printing theta0.shape ", theta0.shape)
print("Printing theta1.shape ", theta1.shape)


# define prediction
yPredicted = np.matmul(x, theta0) + np.matmul(np.ones((x.shape[0], 1), dtype=float), theta1)

#define step
alpha = 0.01

# define cost
cost = np.power((yPredicted-y), 2).sum() / y.shape[0]

def predict(x,theta0, theta1):
    # print("Printing x.shape : ", x.shape)
    # print("Pringing theta1.shape : ", theta1.shape)
    return np.matmul(x, theta0) + np.matmul(np.ones((x.shape[0], 1), dtype=float), theta1)

def gradientDescent(x, diff):
    return np.matmul(x.T, diff)

def costFunction(y, yPredicted):
    return (np.power(y - yPredicted, 2)).sum() / y.shape[0]

for iterations in range(1000000):
    tempTheta0 = theta0 - alpha * gradientDescent(x, (predict(x, theta0, theta1) - y))
    tempTheta1 = theta1 - alpha * np.array([((predict(x, theta0, theta1) - y).sum(axis=0) / y.shape[0]).tolist()])
    # print(np.array([((predict(x, theta0, theta1) - y).sum(axis=0) / y.shape[0]).tolist()]).T.shape)
    # print("Print tempTheta0.shape ",tempTheta0.shape)
    # print("Print tempTheta1.shape ",tempTheta1.shape)
    theta0 = tempTheta0
    theta1 = tempTheta1
    if(iterations%100==0):
        debug = input()
    print("Iteration : ", iterations)
    print("Cost : ", costFunction(y, predict(x, theta0, theta1)), "\n")


print("Theta0 : ",theta0)
print("Theta1 : ",theta1)

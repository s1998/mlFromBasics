import numpy as np

# generating ranom dataset
xInput = np.random.random((100, 2))
yInput = np.zeros((xInput.shape[0], 1))
yInput[np.where(xInput[:,0]+xInput[:,1]>0.5)] = 1
yInput[np.where(xInput[:,0]+xInput[:,1]>1.0)] = 2
yInput[np.where(xInput[:,0]+xInput[:,1]>1.2)] = 3
yInput[np.where(xInput[0]+xInput[1]>1.6)] = 4


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
alpha = 0.001

oututProbab = np.zeros((yInput.shape[0], yInput.max()+1))

for i in range(int(yInput.max()+1)):
    currentClass = i
    yCurrentClass = np.zeros((yInput.shape[0],1))
    yCurrentClass[np.where(yInput[:,0] == currentClass)] = 1
    # for j in range(yCurrentClass.shape[0]):
    #     print(yInput[j], yCurrentClass[j])
    theta = np.zeros((xInput.shape[1]+1, 1))+0.00001
    print(costFunction(yCurrentClass, predict(theta, x)))
    for iterations in range(300000):
        tempTheta = theta - alpha * gradientOfTheta(x, yCurrentClass, theta)
        theta = tempTheta
        if iterations%1000==0:
            print("Iteration number, current class, cost : ", iterations, currentClass, costFunction(yCurrentClass, predict(theta, x)))
    print(currentClass, costFunction(yCurrentClass, predict(theta, x)))
    # print(predict(theta, x))
    # debug = input()
    oututProbab[:, currentClass:currentClass+1] = predict(theta, x)

print(yInput.max()+1)
yPredicted = np.argmax(oututProbab, axis=1).T
for i in range(yInput.shape[0]):
    print(yInput[i], yPredicted[i], oututProbab[i])


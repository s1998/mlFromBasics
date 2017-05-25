import numpy as np

# source : numpy from datacamp.com

# part -1 : Creating arrays
print("\n\n Part-1 : Array creation \n")
a = np.array([1, 2, 3])
b = np.array([ [1,2], [3,4] ], dtype = float)
c = np.array([ [[1,2,3],[4,5,6],[7,8,9]],
               [[10,20,30],[40,50,60],[70,80,90]],
               [[100,200,300],[400,500,600],[700,800,900]]
               ], dtype=float)

print("Printing array a : \n", a, "\n")
print("Prinitng array b : \n", b, "\n")
print("Printing array c : \n", c, "\n")

# part - 2 : Inspecting the array
print("\n\n Part-2 : Array inspection \n")
# a.shape[0] == len(a)
# To work with one-dimensional arrays only, use len(a) to get the array's size eplicitly
# a.ndim -> no of dimensions of the array
# a.size -> total no of elements in the array
# a.shape -> no of elements present in each dimension
# len(a) -> shape[0]

print("\nDetails of a : ")
print(a.ndim)
print(a.size)
print(len(a))
print(a.shape)
print(a.dtype)
print(a.dtype.name)

print("\nDetails of b : ")
print(b.ndim)
print(b.size)
print(len(b))
print(b.shape)
print(b.dtype)
print(b.dtype.name)

print("\nDetails of c : ")
print(c.ndim)
print(c.size)
print(len(c))
print(c.shape)
print(c.dtype)
print(c.dtype.name)

# part - 3 : placeholders initializer (FEZOR)
#  full
#  eye
#  ones
#  zeros
#  random
a = np.zeros((3,3,3), dtype=float)
print("\n Initialized zeros : \n", a)
a = np.eye(3, dtype=float)
print("\n Initialized identity : \n", a)
a = np.full((3,3),7)
print("\n Initialized as full : \n", a)
a = np.ones((3,3,3), dtype=float)
print("\n Initialized as ones : \n", a)
a = np.random.random((3,3,3))
print("\n Initialized as random : \n", a)

# part - 4 : data types
# np.int64
# np.float32
# np.double
# np.complex
# np.bool

# part - 5 : Arithmetic mathematics
# Support for overloading in numpy
# vector - matrix
# matrix - matrix

a = np.array([1,2,3])
b = np.array([[3,2,3], [4,2,6], [7,8,1]])
c = np.eye(3)

# vector - matrix
print("a-b\n",a-b)
print("a-c\n",a-c)
# matrix - matrix
print("b-c\n",b-c)

# vector - matrix
print("a*b\n",a*b)
print("a*c\n",a*c)
# matrix - matrix
print("b*c\n",b*c)
print("b*c\n", np.matmul(b,c))

# elment wise operation
print("Exponent :\n",np.exp(c))
print("Sqrt :\n",np.sqrt(c))
print("Sin :\n",np.sin(c))
print("Cos :\n",np.cos(c))
print("Log :\n",np.log(c+1))

print("Sum np.sum(a) :\n", np.sum(a))
print("Sum np.sum(b) :\n", np.sum(b))
print("Sum np.sum(b) :\n", np.sum(c))
print("Min np.min(a) :\n", np.min(a))
print("Min np.min(b) :\n", np.min(b))
print("Min np.min(c) :\n", np.min(c))

print("Min a.min() :\n", a.min())
print("Min a.min() :\n", b.min())
print("Min c.min() :\n", c.min())

# proces column wise for axis = 0 and row wise for axis = 1
print("Min b.min(axis = 0) :\n", b.min(axis=0))
temp = b.sum(axis=0)
print("Min b.sum(axis = 0) :\n", temp)
temp = np.array([temp.tolist()]).T
print(temp.shape)
print("Min b.min(axis = 1) :\n", b.min(axis=1))

print("Comparing matrices a==b: \n", a==b)
print("Comparing matrices b==c: \n", c==b)
print("Comparing matrices b<c: \n", b<c)
print("Comparing matrices b<4: \n", b<4)

a.sort()
print("Sorted version of a :\n", a)
# Creating a copy
copyOfA = a.copy()

tempB = b.copy()
tempB.sort(axis=0)
print("Sorted version of b col-wise : \n", tempB)
tempB = b.copy()
tempB.sort(axis=1)
print("Sorted version of b row-wise : \n", tempB)
tempB = b
tempB.sort(axis=1)
print("Sorting tempB = b row-wise and then printing b : \n", b)

print(" proces column wise for axis = 0 and row wise for axis = 1")
print(" b = np.array([[1,2],[2,3]]) " )
print(" b, np.sum(b, axis = 0) is [3 3] " )
print(" b, np.sum(b, axis = 1) is [2 4] " )
b = np.array([[1,1],[2,2]])
print(np.sum(b, axis = 0))
print(np.sum(b, axis = 1))

print(max(0, 1e-300))


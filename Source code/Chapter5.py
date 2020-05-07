


# Load library
import numpy as np
# Create a vector as a row
vector_row = np.array([1, 2, 3])
# Create a vector as a column
vector_column = np.array([[1],
[2],
[3]])

# Load library
import numpy as np
# Create a matrix
matrix = np.array([[1, 2],
[1, 2],
[1, 2]])

matrix_object = np.mat([[1, 2],
[1, 2],
[1, 2]])
matrix([[1, 2],
[1, 2],
[1, 2]])


# Load libraries
import numpy as np
from scipy import sparse
# Create a matrix
matrix = np.array([[0, 0],
[0, 1],
[3, 0]])
# Create compressed sparse row (CSR) matrix
matrix_sparse = sparse.csr_matrix(matrix)


# View sparse matrix
print(matrix_sparse)
(1, 1) 1
(2, 0) 3


# Create larger matrix
matrix_large = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# Create compressed sparse row (CSR) matrix
matrix_large_sparse = sparse.csr_matrix(matrix_large)
# View original sparse matrix
print(matrix_sparse)
(1, 1) 1
(2, 0) 3
# View larger sparse matrix
print(matrix_large_sparse)
(1, 1) 1
(2, 0) 3


# Load library
import numpy as np
# Create row vector
vector = np.array([1, 2, 3, 4, 5, 6])
# Create matrix
matrix = np.array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
# Select third element of vector
vector[2]
3
# Select second row, second column
matrix[1,1]
5


# Select all elements of a vector
vector[:]
array([1, 2, 3, 4, 5, 6])
# Select everything up to and including the third element
vector[:3]
array([1, 2, 3])
# Select everything after the third element
vector[3:]
array([4, 5, 6])
# Select the last element
vector[-1]
6
# Select the first two rows and all columns of a matrix
matrix[:2,:]
array([[1, 2, 3],
[4, 5, 6]])
# Select all rows and the second column
matrix[:,1:2]
array([[2],
[5],
[8]])


# Load library
import numpy as np
# Create matrix
matrix = np.array([[1, 2, 3, 4],
[5, 6, 7, 8],
[9, 10, 11, 12]])
# View number of rows and columns
matrix.shape
(3, 4)
# View number of elements (rows * columns)
matrix.size
12
# View number of dimensions
matrix.ndim
2


# Load library
import numpy as np
# Create matrix
matrix = np.array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
# Create function that adds 100 to something
add_100 = lambda i: i + 100
# Create vectorized function
vectorized_add_100 = np.vectorize(add_100)
# Apply function to all elements in matrix
vectorized_add_100(matrix)
array([[101, 102, 103],
[104, 105, 106],
[107, 108, 109]])


# Add 100 to all elements
matrix + 100
array([[101, 102, 103],
[104, 105, 106],
[107, 108, 109]])


# Load library
import numpy as np
# Create matrix
matrix = np.array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
# Return maximum element
np.max(matrix)
9
# Return minimum element
np.min(matrix)
1


# Find maximum element in each column
np.max(matrix, axis=0)
array([7, 8, 9])
# Find maximum element in each row
np.max(matrix, axis=1)
array([3, 6, 9])


# Load library
import numpy as np
# Create matrix
matrix = np.array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
# Return mean
np.mean(matrix)
5.0
# Return variance
np.var(matrix)
6.666666666666667
# Return standard deviation
np.std(matrix)
2.5819888974716112


# Find the mean value in each column
np.mean(matrix, axis=0)
array([ 4., 5., 6.])


# Load library
import numpy as np
# Create 4x3 matrix
matrix = np.array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9],
[10, 11, 12]])
# Reshape matrix into 2x6 matrix
matrix.reshape(2, 6)
array([[ 1, 2, 3, 4, 5, 6],
[ 7, 8, 9, 10, 11, 12]])
1.9


# Load library
import numpy as np
# Create matrix
matrix = np.array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
# Transpose matrix
matrix.T
array([[1, 4, 7],
[2, 5, 8],
[3, 6, 9]])


# Transpose vector
np.array([1, 2, 3, 4, 5, 6]).T
array([1, 2, 3, 4, 5, 6])


# Tranpose row vector
np.array([[1, 2, 3, 4, 5, 6]]).T
array([[1],
[2],
[3],
[4],
[5],
[6]])


# Load library
import numpy as np
# Create matrix
matrix = np.array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
# Flatten matrix
matrix.flatten()
array([1, 2, 3, 4, 5, 6, 7, 8, 9])


Load library
import numpy as np
# Create matrix
matrix = np.array([[1, 1, 1],
[1, 1, 10],
[1, 1, 15]])
# Return matrix rank
np.linalg.matrix_rank(matrix)
2


# Load library
import numpy as np
# Create matrix
matrix = np.array([[1, 2, 3],
[2, 4, 6],
[3, 8, 9]])
# Return determinant of matrix
np.linalg.det(matrix)
0.0


# Load library
import numpy as np
# Create matrix
matrix = np.array([[1, 2, 3],
[2, 4, 6],
[3, 8, 9]])
# Return diagonal elements
matrix.diagonal()
array([1, 4, 9])


# Return diagonal one above the main diagonal
matrix.diagonal(offset=1)
array([2, 6])
# Return diagonal one below the main diagonal
matrix.diagonal(offset=-1)
array([2, 8])



# Load library
import numpy as np
# Create matrix
matrix = np.array([[1, 2, 3],
[2, 4, 6],
[3, 8, 9]])
# Return trace
matrix.trace()
14


# Return diagonal and sum elements
sum(matrix.diagonal())
14



# Load library
import numpy as np
# Create matrix
matrix = np.array([[1, -1, 3],
[1, 1, 6],
[3, 8, 9]])
# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)
# View eigenvalues
eigenvalues
array([ 13.55075847, 0.74003145, -3.29078992])
# View eigenvectors
eigenvectors
array([[-0.17622017, -0.96677403, -0.53373322],
[-0.435951 , 0.2053623 , -0.64324848],
[-0.88254925, 0.15223105, 0.54896288]])


# Load library
import numpy as np
# Create two vectors
vector_a = np.array([1,2,3])
vector_b = np.array([4,5,6])
# Calculate dot product
np.dot(vector_a, vector_b)
32


# Load library
import numpy as np
# Create matrix
matrix_a = np.array([[1, 1, 1],
[1, 1, 1],
[1, 1, 2]])
# Create matrix
matrix_b = np.array([[1, 3, 1],
[1, 3, 1],
[1, 3, 8]])
# Add two matrices
np.add(matrix_a, matrix_b)
array([[ 2, 4, 2],
[ 2, 4, 2],
[ 2, 4, 10]])
# Subtract two matrices
np.subtract(matrix_a, matrix_b)
array([[ 0, -2, 0],
[ 0, -2, 0],
[ 0, -2, -6]])


# Add two matrices
matrix_a + matrix_b
array([[ 2, 4, 2],
[ 2, 4, 2],
[ 2, 4, 10]])


# Load library
import numpy as np
# Create matrix
matrix_a = np.array([[1, 1],
[1, 2]])
# Create matrix
matrix_b = np.array([[1, 3],
[1, 2]])
# Multiply two matrices
np.dot(matrix_a, matrix_b)
array([[2, 5],
[3, 7]])


# Multiply two matrices
matrix_a @ matrix_b
array([[2, 5],
[3, 7]])


# Multiply two matrices element-wise
matrix_a * matrix_b
array([[1, 3],
[1, 4]])


# Load library
import numpy as np
# Create matrix
matrix = np.array([[1, 4],
[2, 5]])
# Calculate inverse of matrix
np.linalg.inv(matrix)
array([[-1.66666667, 1.33333333],
[ 0.66666667, -0.33333333]])


# Multiply matrix and its inverse
matrix @ np.linalg.inv(matrix)
array([[ 1., 0.],
[ 0., 1.]])


# Load library
import numpy as np
# Set seed
np.random.seed(0)
# Generate three random floats between 0.0 and 1.0
np.random.random(3)
array([ 0.5488135 , 0.71518937, 0.60276338])


# Generate three random integers between 1 and 10
np.random.randint(0, 11, 3)
array([3, 7, 9])

# Draw three numbers from a normal distribution with mean 0.0
# and standard deviation of 1.0
np.random.normal(0.0, 1.0, 3)
array([-1.42232584, 1.52006949, -0.29139398])
# Draw three numbers from a logistic distribution with mean 0.0 and scale of 1.0
np.random.logistic(0.0, 1.0, 3)
array([-0.98118713, -0.08939902, 1.46416405])
# Draw three numbers greater than or equal to 1.0 and less than 2.0
np.random.uniform(1.0, 2.0, 3)
array([ 1.47997717, 1.3927848 , 1.83607876])
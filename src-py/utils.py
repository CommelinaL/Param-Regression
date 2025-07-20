import numpy as np
import struct
import os

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def normalize(x, eval_gradient = False):
     # Calculate the sum along the specified axis
    s = np.sum(x, axis=1, keepdims=True)
    epsilon = 1e-12
    x_n = x / (s + epsilon)
    if eval_gradient == False:
        return x_n
    # Calculate the gradient
    # Initialize the gradient matrix
    gradient = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):  # For each sample
        for j in range(x.shape[1]):  # For each output dimension
            for k in range(x.shape[1]): # For each input dimension
                gradient[i, j, k] = (s[i][0] * (1 if j == k else 0) - x[i, k]) / (s[i][0] ** 2 + epsilon)
    return x_n, gradient

def read_vector_from_file(filename):
    with open(filename, 'rb') as inFile:
        # Read the size of the vector
        size_bytes = inFile.read(4)
        size = struct.unpack('I', size_bytes)[0]

        # Read the vector data
        data_bytes = inFile.read(size * 8)  # Each double is 8 bytes
        data = np.frombuffer(data_bytes, dtype=np.float64)  # Read as numpy array of doubles
    return data[:size]  # Return only the read elements

def write_vector_to_file(filename, vector):
    # Check if the directory exists
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Ensure the input is a NumPy array
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector, dtype=np.float64)
    
    # Ensure the vector is 1-dimensional
    if vector.ndim != 1:
        raise ValueError("Input must be a 1-dimensional vector")
    
    # Get the size of the vector
    size = len(vector)
    
    with open(filename, 'wb') as outFile:
        # Write the size of the vector (4 bytes, unsigned int)
        outFile.write(struct.pack('I', size))
        
        # Write the vector data (each element as 8-byte double)
        vector.astype(np.float64).tofile(outFile)

def read_pts_from_file(filename):
    points = []
    with open(filename, 'r') as point_file:
        point_num = int(point_file.readline())
        for _ in range(point_num):
            point_str = point_file.readline()
            points += [float(x) for x in point_str.split()]
    return points


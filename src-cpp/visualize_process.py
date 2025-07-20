import struct
import numpy as np
import matplotlib.pyplot as plt

def read_vector_from_file(filename):
    with open(filename, 'rb') as inFile:
        # Read the size of the vector
        size_bytes = inFile.read(4)
        size = struct.unpack('I', size_bytes)[0]

        # Read the vector data
        data_bytes = inFile.read(size * 8)  # Each double is 8 bytes
        data = np.frombuffer(data_bytes, dtype=np.float64)  # Read as numpy array of doubles

    return data[:size]  # Return only the read elements

def polyline_from_file(best_file, label="Best"):
    vector = read_vector_from_file(best_file)

    # Generate x values for the polylines
    x_values = np.arange(len(vector))  # Assuming both vectors have the same length

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the polyline
    plt.plot(x_values, vector, marker='o', linestyle='-', color='b', label=label) 

    # Adding titles and labels
    plt.title('Optimization')
    plt.xlabel('Generation')
    plt.ylabel('Cost')

    # Adding a legend to distinguish the two polylines
    plt.legend()

    # Adding grid for better readability
    plt.grid()

    # Show the graph
    plt.show()

if __name__ == "__main__":
    polyline_from_file(r"E:\BSplineLearning\src-cpp\B-spline-curve-fitting\GA\0\best_record.bin")
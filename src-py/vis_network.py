import matplotlib.pyplot as plt

def plot_mlp():
    plt.figure(figsize=(10, 6))

    # Define the number of layers and neurons
    layers = [3, 100, 100, 1]  # Example: 3 input features, 2 hidden layers with 100 neurons each, 1 output
    layer_sizes = len(layers)

    for layer in range(layer_sizes):
        for neuron in range(layers[layer]):
            # Plot each neuron
            plt.scatter(layer, neuron, s=1000, c='blue' if layer < layer_sizes - 1 else 'orange', edgecolor='black')

            # Connect to previous layer
            if layer > 0:
                for prev_neuron in range(layers[layer - 1]):
                    plt.plot([layer - 1, layer], [prev_neuron, neuron], c='grey', alpha=0.5)

    # Set axis limits and labels
    plt.xlim(-0.5, layer_sizes - 0.5)
    plt.ylim(-1, max(layers) + 1)
    plt.xticks(range(layer_sizes), ['Input Layer', 'Hidden Layer 1', 'Hidden Layer 2', 'Output Layer'])
    plt.title('MLP Network Structure')
    plt.show()

plot_mlp()
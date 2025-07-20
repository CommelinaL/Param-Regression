import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to read the CSV file and plot the specified columns
def plot_metrics_with_two_y_axes(csv_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Extract the relevant columns
    rmse = data['RMSE (normalized)']
    medae = data['MedAE (normalized)']
    r2 = data['R2 (normalized)']

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot RMSE and MedAE on the first Y-axis
    x = np.arange(20000, 200001, 20000)
    ax1.plot(x, rmse, label='RMSE', color='blue', marker='o')
    ax1.plot(x, medae, label='MedAE', color='red', marker='s')
    ax1.plot(140000, rmse[6], 'b*')
    ax1.text(140000, rmse[6], "(140000, {:.4})".format(rmse[6]))
    ax1.plot(140000, medae[6], 'r*')
    ax1.text(140000, medae[6], "(140000, {:.4})".format(medae[6]))
    ax1.set_xlabel('Size of training dataset')
    ax1.set_ylabel('RMSE / MedAE', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')
    
    # Create a second Y-axis for R2
    ax2 = ax1.twinx()
    ax2.plot(x, r2, label='R2', color='green', marker='^')
    ax2.plot(140000, r2[6], 'g*')
    ax2.text(140000, r2[6], "(140000, {:.4})".format(r2[6]))
    ax2.set_ylabel('R2', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Add title and grid
    #plt.title('Influence of dataset size')
    ax1.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()

# Function to read the CSV file and plot the specified columns
def plot_metrics(csv_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Extract the relevant columns
    rmse = data['RMSE (normalized)']
    medae = data['MedAE (normalized)']
    r2 = data['R2 (normalized)']

    # Create a figure and axis
    plt.figure(figsize=(10, 6))

    # Plot each metric
    x = np.arange(20000, 200001, 20000)
    plt.plot(x, rmse, label='RMSE', marker='o')
    plt.plot(x, medae, label='MedAE', marker='s')
    plt.plot(x, r2, label='R2', marker='^')

    plt.plot(140000, rmse[6], 'r*')
    plt.text(140000, rmse[6], "(140000, {:.4})".format(rmse[6]))
    plt.plot(140000, medae[6], 'b*')
    plt.text(140000, medae[6], "(140000, {:.4})".format(medae[6]))
    plt.plot(140000, r2[6], 'g*')
    plt.text(140000, r2[6], "(140000, {:.4})".format(r2[6]))

    # Add labels and title
    plt.xlabel('Size of training dataset')
    plt.ylabel('Regression metrics')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage
csv_file_path = 'dataset_test.csv'  # Replace with your CSV file path
plot_metrics_with_two_y_axes(csv_file_path)

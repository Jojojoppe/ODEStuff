import matplotlib.pyplot as plt
import numpy as np
import sys

def read_data(filename):
    """Reads a space-separated text file and returns the time and solution arrays."""
    data = np.loadtxt(filename)
    t = data[:, 0]  # First column is time
    y = data[:, 1:]  # Remaining columns are solution values
    return t, y

def plot_results(filenames):
    """Plots results from multiple files, creating a subplot for each column."""
    
    # Read all data first to determine the maximum number of variables
    all_data = []
    max_vars = 0

    for filename in filenames:
        try:
            t, y = read_data(filename)
            num_vars = y.shape[1] if len(y.shape) > 1 else 1  # Handle scalar vs vector case
            all_data.append((filename, t, y, num_vars))
            max_vars = max(max_vars, num_vars)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Create subplots
    fig, axes = plt.subplots(max_vars, 1, figsize=(8, 4 * max_vars), sharex=True)
    if max_vars == 1:
        axes = [axes]  # Ensure axes is iterable if only one subplot

    for filename, t, y, num_vars in all_data:
        for i in range(num_vars):
            label = f"{filename} (var {i+1})"
            axes[i].plot(t, y[:, i] if num_vars > 1 else y, label=label, marker="o", markersize=3, linestyle="--")
            axes[i].set_ylabel(f"Variable {i+1}")
            axes[i].legend()
            axes[i].grid()

    axes[-1].set_xlabel("Time")  # Only the last subplot gets an x-label
    fig.suptitle("ODE Solver Results")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py results_fe.txt results_rk4.txt [...]")
    else:
        plot_results(sys.argv[1:])

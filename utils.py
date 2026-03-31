# utils.py
import numpy as np
import matplotlib.pyplot as plt

# Helper function to visualize patterns
def plot_pattern(pattern, title, shape=(5, 5)):
    plt.imshow(pattern.reshape(shape), cmap='binary')
    plt.title(title)
    plt.axis('off')

# Calculate Hamming distance between two patterns
def hamming_distance(pattern1, pattern2):
    return np.sum(pattern1 != pattern2)
# main.py
import numpy as np
import matplotlib.pyplot as plt

# Local module imports
from logic import HopfieldNetwork
from utils import plot_pattern, hamming_distance
from patterns import (
    pattern_T, pattern_H, pattern_I, patterns,
    pattern_T_large, pattern_H_large, pattern_I_large, patterns_large
)

def run_5x5_network():
    # Visualize the original patterns
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plot_pattern(pattern_T, "Original T")
    plt.subplot(1, 3, 2)
    plot_pattern(pattern_H, "Original H")
    plt.subplot(1, 3, 3)
    plot_pattern(pattern_I, "Original I")
    plt.tight_layout()
    plt.show()

    # Initialize the Hopfield Network (25 neurons for 5x5 patterns)
    num_neurons = patterns[0].size
    hopfield_net = HopfieldNetwork(num_neurons)

    # Train the network with the patterns
    hopfield_net.train(patterns)
    print("Hopfield Network trained successfully with 3 patterns (T, H, I).")

    # Create noisy versions of patterns
    noisy_T = np.copy(pattern_T)
    noisy_T[4] *= -1 
    noisy_T[10] *= -1 
    noisy_T[13] *= -1 

    noisy_H = np.copy(pattern_H)
    noisy_H[1] *= -1
    noisy_H[7] *= -1

    noisy_I = np.copy(pattern_I)
    noisy_I[3] *= -1
    noisy_I[20] *= -1

    # Recall the patterns from the noisy input
    recalled_T = hopfield_net.recall(noisy_T)
    recalled_H = hopfield_net.recall(noisy_H)
    recalled_I = hopfield_net.recall(noisy_I)

    # Visualize the noisy and recalled patterns
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plot_pattern(noisy_T, "Noisy T")
    plt.subplot(2, 3, 4)
    plot_pattern(recalled_T, "Recalled T")

    plt.subplot(2, 3, 2)
    plot_pattern(noisy_H, "Noisy H")
    plt.subplot(2, 3, 5)
    plot_pattern(recalled_H, "Recalled H")

    plt.subplot(2, 3, 3)
    plot_pattern(noisy_I, "Noisy I")
    plt.subplot(2, 3, 6)
    plot_pattern(recalled_I, "Recalled I")

    plt.tight_layout()
    plt.show()

    # Check if the recalled patterns match the original ones
    print(f"Does recalled 'T' match original 'T'? {np.array_equal(recalled_T, pattern_T)}")
    print(f"Does recalled 'H' match original 'H'? {np.array_equal(recalled_H, pattern_H)}")
    print(f"Does recalled 'I' match original 'I'? {np.array_equal(recalled_I, pattern_I)}\n")
    
    return noisy_T, recalled_T

def run_7x7_network():
    # Visualize the larger original patterns
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plot_pattern(pattern_T_large, "Original Large T", shape=(7,7))
    plt.subplot(1, 3, 2)
    plot_pattern(pattern_H_large, "Original Large H", shape=(7,7))
    plt.subplot(1, 3, 3)
    plot_pattern(pattern_I_large, "Original Large I", shape=(7,7))
    plt.tight_layout()
    plt.show()

    # Initialize the Hopfield Network with 49 neurons
    num_neurons_large = patterns_large[0].size
    hopfield_net_large = HopfieldNetwork(num_neurons_large)

    # Train the network with the larger patterns
    hopfield_net_large.train(patterns_large)
    print("Hopfield Network (large) trained successfully with 3 patterns (Large T, H, I).")

    # Create noisy versions of the large patterns
    noisy_T_large = np.copy(pattern_T_large)
    noisy_T_large[5] *= -1 
    noisy_T_large[12] *= -1 
    noisy_T_large[20] *= -1 
    noisy_T_large[25] *= -1 
    noisy_T_large[30] *= -1 

    noisy_H_large = np.copy(pattern_H_large)
    noisy_H_large[2] *= -1
    noisy_H_large[10] *= -1
    noisy_H_large[15] *= -1

    noisy_I_large = np.copy(pattern_I_large)
    noisy_I_large[4] *= -1
    noisy_I_large[28] *= -1
    noisy_I_large[35] *= -1

    # Recall the patterns from the noisy inputs
    recalled_T_large = hopfield_net_large.recall(noisy_T_large)
    recalled_H_large = hopfield_net_large.recall(noisy_H_large)
    recalled_I_large = hopfield_net_large.recall(noisy_I_large)

    # Visualize the noisy and recalled large patterns
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plot_pattern(noisy_T_large, "Noisy Large T", shape=(7,7))
    plt.subplot(2, 3, 4)
    plot_pattern(recalled_T_large, "Recalled Large T", shape=(7,7))

    plt.subplot(2, 3, 2)
    plot_pattern(noisy_H_large, "Noisy Large H", shape=(7,7))
    plt.subplot(2, 3, 5)
    plot_pattern(recalled_H_large, "Recalled Large H", shape=(7,7))

    plt.subplot(2, 3, 3)
    plot_pattern(noisy_I_large, "Noisy Large I", shape=(7,7))
    plt.subplot(2, 3, 6)
    plot_pattern(recalled_I_large, "Recalled Large I", shape=(7,7))

    plt.tight_layout()
    plt.show()

    # Check if the recalled patterns match the original ones
    print(f"Does recalled 'Large T' match original 'Large T'? {np.array_equal(recalled_T_large, pattern_T_large)}")
    print(f"Does recalled 'Large H' match original 'Large H'? {np.array_equal(recalled_H_large, pattern_H_large)}")
    print(f"Does recalled 'Large I' match original 'Large I'? {np.array_equal(recalled_I_large, pattern_I_large)}\n")

def run_analysis(noisy_T, recalled_T):
    print("Analysis for 'T' pattern:")

    # Visualize the original, noisy, and recalled 'T' patterns
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plot_pattern(pattern_T, "Original T")

    plt.subplot(1, 3, 2)
    plot_pattern(noisy_T, "Noisy T")

    plt.subplot(1, 3, 3)
    plot_pattern(recalled_T, "Recalled T")

    plt.tight_layout()
    plt.show()

    hd_noisy_T = hamming_distance(pattern_T, noisy_T)
    hd_recalled_T = hamming_distance(pattern_T, recalled_T)

    print(f"Hamming distance between Original T and Noisy T: {hd_noisy_T} bits")
    print(f"Hamming distance between Original T and Recalled T: {hd_recalled_T} bits")

    # Display the actual patterns to see where the mismatch occurs
    print("\nOriginal T:")
    print(pattern_T.reshape(5,5))
    print("\nNoisy T:")
    print(noisy_T.reshape(5,5))
    print("\nRecalled T:")
    print(recalled_T.reshape(5,5))

    # Highlight differences
    diff_T = (pattern_T != recalled_T).astype(int)
    print("\nDifferences (1 where mismatch, 0 where match):")
    print(diff_T.reshape(5,5))

if __name__ == "__main__":
    # Execute 5x5 network logic and store state for analysis
    noisy_T_state, recalled_T_state = run_5x5_network()
    
    # Execute 7x7 network logic
    run_7x7_network()
    
    # Run analysis block based on 5x5 T pattern results
    run_analysis(noisy_T_state, recalled_T_state)
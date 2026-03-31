# logic.py
import numpy as np

class HopfieldNetwork:
    # Initialize the network with a specified number of neurons
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    # Train the network using Hebbian learning
    def train(self, patterns):
        # Patterns should be bipolar (-1 or 1)
        for p in patterns:
            # Reshape pattern to be a column vector
            p = p.reshape(-1, 1)
            # Hebbian learning rule
            self.weights += p @ p.T
        
        # Set diagonal elements to zero (no self-connections)
        np.fill_diagonal(self.weights, 0)

    # Recall a pattern from an input state
    def recall(self, pattern, max_iterations=100):
        # Pattern should be bipolar (-1 or 1)
        state = np.copy(pattern)
        for _ in range(max_iterations):
            prev_state = np.copy(state)
            # Asynchronous update
            for i in range(self.num_neurons):
                # Calculate activation
                activation = self.weights[i, :] @ state
                # Apply sign activation function
                state[i] = 1 if activation >= 0 else -1
            
            # If the state stabilizes, stop iterating
            if np.array_equal(state, prev_state):
                break
        return state
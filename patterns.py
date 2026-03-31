# patterns.py
import numpy as np

# Define some example patterns (5x5 images, flattened to 25 neurons)
# Representing simple shapes like 'T', 'H', 'I'

# 'T' pattern
pattern_T = np.array([
    1,  1,  1,  1,  1,
    -1, 1,  -1, 1, -1,
    -1, 1,  -1, 1, -1,
    -1, 1,  -1, 1, -1,
    -1, 1,  -1, 1, -1
])

# 'H' pattern
pattern_H = np.array([
    1, -1, 1, -1, 1,
    1, -1, 1, -1, 1,
    1,  1, 1,  1, 1,
    1, -1, 1, -1, 1,
    1, -1, 1, -1, 1
])

# 'I' pattern
pattern_I = np.array([
    1,  1,  1,  1,  1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    1,  1,  1,  1,  1
])

patterns = [pattern_T, pattern_H, pattern_I]

# Define new larger example patterns (7x7 images, flattened to 49 neurons)

# 'T' pattern (7x7)
pattern_T_large = np.array([
    1,  1,  1,  1,  1,  1,  1,
    -1, -1, 1,  1,  1, -1, -1,
    -1, -1, 1,  1,  1, -1, -1,
    -1, -1, 1,  1,  1, -1, -1,
    -1, -1, 1,  1,  1, -1, -1,
    -1, -1, 1,  1,  1, -1, -1,
    -1, -1, 1,  1,  1, -1, -1
])

# 'H' pattern (7x7)
pattern_H_large = np.array([
    1, -1, -1, 1, -1, -1, 1,
    1, -1, -1, 1, -1, -1, 1,
    1, -1, -1, 1, -1, -1, 1,
    1,  1,  1, 1,  1,  1, 1,
    1, -1, -1, 1, -1, -1, 1,
    1, -1, -1, 1, -1, -1, 1,
    1, -1, -1, 1, -1, -1, 1
])

# 'I' pattern (7x7)
pattern_I_large = np.array([
    1,  1,  1,  1,  1,  1,  1,
    -1, -1, -1, 1, -1, -1, -1,
    -1, -1, -1, 1, -1, -1, -1,
    -1, -1, -1, 1, -1, -1, -1,
    -1, -1, -1, 1, -1, -1, -1,
    -1, -1, -1, 1, -1, -1, -1,
    1,  1,  1,  1,  1,  1,  1
])

patterns_large = [pattern_T_large, pattern_H_large, pattern_I_large]
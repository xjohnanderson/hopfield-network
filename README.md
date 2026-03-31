# README.md
# Hopfield Network Implementation

A modular, clean-code implementation of a Hopfield Network capable of memorizing and recalling 5x5 and 7x7 bipolar image patterns. The system demonstrates associative memory by injecting noise into standard patterns ('T', 'H', 'I') and recovering the original states.

## Architecture
- **logic.py**: Contains the core `HopfieldNetwork` business logic and Hebbian learning implementation.
- **utils.py**: Includes visual plotting functions and pure mathematical operations (Hamming distance).
- **patterns.py**: Acts as a data store for the static 5x5 and 7x7 input arrays.
- **main.py**: The entry point that orchestrates network instantiation, training, testing, and analysis.

## Usage

1. **Install dependencies:**
   Ensure you have an environment configured with the requirements.
   ```bash
   pip install -r requirements.txt
2. **Run the network:**
    Execute the main script. The script will render plots sequentially. Close each plot window to proceed to the next stage of execution.

    '''bash
    python main.py
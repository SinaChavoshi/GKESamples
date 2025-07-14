import numpy as np

# --- Configuration Constants (from the training script) ---
NUM_DENSE_FEATURES = 13
VOCAB_SIZES = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
    40000000, 40000000, 590152, 12973, 108, 36
]
NUM_SPARSE_FEATURES = len(VOCAB_SIZES)

def generate_data(filename, num_samples):
    """Generates synthetic ranking data and saves it to a TSV file."""
    print(f"Generating {num_samples} samples for {filename}...")
    
    with open(filename, "w") as f:
        for _ in range(num_samples):
            # Generate a random label (0 or 1)
            label = np.random.randint(0, 2)
            
            # Generate random dense features (floats between 0 and 1)
            dense_features = np.random.rand(NUM_DENSE_FEATURES)
            
            # Generate random sparse features (integers within each vocab size)
            sparse_features = [np.random.randint(0, vocab_size) for vocab_size in VOCAB_SIZES]
            
            # Combine all features into a single, tab-separated line
            all_features = [label] + list(dense_features) + list(sparse_features)
            line = "\t".join(map(str, all_features))
            
            f.write(line + "\n")
            
    print(f"Successfully created {filename}.")

if __name__ == "__main__":
    # Generate a small training and evaluation set
    # Using a smaller number of samples for a quick test
    generate_data("train_sample.tsv", num_samples=100000)
    generate_data("eval_sample.tsv", num_samples=20000)


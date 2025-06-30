import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer

def main(args):
    # Authenticate with Hugging Face
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found in environment variable HF_TOKEN")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)

    # Load the raw dataset from GCS
    raw_dataset = load_dataset("json", data_files=args.input_path, split="train")

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_seq_length)

    print("Tokenizing and mapping the dataset...")
    processed_dataset = raw_dataset.map(tokenize_function, batched=True)

    # Remove the original text column
    processed_dataset = processed_dataset.remove_columns(["text"])

    # Set the format for PyTorch
    processed_dataset.set_format("torch")

    # Save the processed dataset back to GCS
    print(f"Saving processed dataset to {args.output_path}...")
    processed_dataset.save_to_disk(args.output_path)
    print("Dataset preprocessing complete and saved to GCS.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="The model ID for the tokenizer.")
    parser.add_argument("--input_path", type=str, required=True, help="The GCS path to the raw training_data.jsonl.")
    parser.add_argument("--output_path", type=str, required=True, help="The GCS path to save the processed dataset.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length for tokenization.")
    args = parser.parse_args()
    main(args)

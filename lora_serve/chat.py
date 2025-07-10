import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel # Make sure to import PeftModel

def main():
    """Main function to run the interactive chat."""
    parser = argparse.ArgumentParser(description="Chat with a LoRa fine-tuned model.")
    # Path to the directory containing the LoRa adapter
    parser.add_argument("--model_path", type=str, required=True, help="The local filesystem path to the LoRa adapter directory.")
    # Add an argument for the base model ID, with a default value
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="The base model ID from Hugging Face.")
    args = parser.parse_args()

    # --- 1. Load Tokenizer and Base Model ---
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found in environment variable HF_TOKEN")

    print(f"Loading tokenizer from base model: {args.base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, token=hf_token)

    print(f"Loading base model: {args.base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # --- 2. Load and Attach LoRa Adapter ---
    print(f"Loading LoRa adapter from local path: {args.model_path}")
    # Use PeftModel to apply the adapter to the base model
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()

    # --- 3. Start Interactive Chat Loop (no changes needed here) ---
    print("\n\nModel loaded. Start chatting! Type 'quit' or 'exit' to end the session.")
    print("=======================================================================")
    while True:
        user_input = input("### Human: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        prompt = f"### Human: {user_input} ### Assistant:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = response_text.split("### Assistant:")[-1].strip()
        print(f"### Assistant: {assistant_response}\n")

if __name__ == "__main__":
    main()

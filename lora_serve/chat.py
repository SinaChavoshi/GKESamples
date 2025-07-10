#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    """Main function to run the interactive chat."""
    parser = argparse.ArgumentParser(description="Chat with a fine-tuned LoRA model.")
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="The base model ID from Hugging Face.")
    parser.add_argument("--lora_adapter_path", type=str, required=True, help="The local filesystem path to the trained LoRA adapter directory.")
    args = parser.parse_args()

    # --- 1. Load Tokenizer and Base Model ---
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found in environment variable HF_TOKEN")

    print(f"Loading tokenizer for base model: {args.base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, token=hf_token)

    print(f"Loading base model: {args.base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # --- 2. Load and Merge the LoRA Adapter from the FUSE path ---
    print(f"Loading and applying LoRA adapter from path: {args.lora_adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
    model.eval()

    # --- 3. Start Interactive Chat Loop ---
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
                max_new_tokens=256,
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

import argparse
import os
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def main(args):
    # Load dataset from GCS
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # Load model and tokenizer
    # Note: Using bfloat16 for TPU compatibility
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        use_auth_token=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_id,
        use_auth_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # Add LoRA adapters to the model
    model = get_peft_model(model, lora_config)

    # Define training arguments
    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        save_strategy="epoch",
        bf16=True, # Use bfloat16 for TPUs
        push_to_hub=False,
    )

    # Initialize the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        packing=True,
        peft_config=lora_config,
    )

    # Start training
    print("Starting LoRA fine-tuning...")
    trainer.train()
    print("Training complete.")

    # Save the final adapter
    final_output_dir = os.path.join(args.output_dir, "final_checkpoint")
    trainer.save_model(final_output_dir)
    print(f"LoRA adapter saved to {final_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3-8B-Instruct", help="The model ID from Hugging Face.")
    parser.add_argument("--dataset_path", type=str, required=True, help="The GCS path to the training_data.jsonl file.")
    parser.add_argument("--output_dir", type=str, required=True, help="The GCS path to save the output adapter.")
    args = parser.parse_args()
    main(args)

import argparse
import os
import torch
import transformers
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def main(args):
    # Get the Hugging Face token from an environment variable
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found in environment variable HUGGING_FACE_HUB_TOKEN")

    print("Loading pre-processed dataset from GCS...")
    processed_dataset = load_from_disk(args.dataset_path)
    print("Dataset loaded successfully.")

    print(f"Loading base model: {args.model_id}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_id,
        token=hf_token,
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("Base model and tokenizer loaded successfully.")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("LoRA adapters added to the model.")

    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        save_strategy="epoch",
        bf16=True,
        push_to_hub=False,
        report_to="none",
    )

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )
    print("SFTTrainer initialized successfully.")

    print("Starting LoRA fine-tuning...")
    trainer.train()
    print("Training complete.")

    final_output_dir = os.path.join(args.output_dir, "final_checkpoint")
    print(f"Saving final adapter to {final_output_dir}...")
    trainer.save_model(final_output_dir)
    print(f"LoRA adapter saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="The model ID from Hugging Face.")
    parser.add_argument("--dataset_path", type=str, required=True, help="The GCS path to the *processed* dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="The GCS path to save the output adapter.")
    args = parser.parse_args()
    main(args)

# Author: Alden Porter
# Date created: 2024-06-15

# This script is designed to train a language model using the MATA framework. It includes functions for model training, and evaluation. The script is structured to allow for easy customization of hyperparameters and model architecture. Oddly, that was written by AI. Who knows how a parameter becomes hyper? Too much caffine perhaps.

#---------------------------------------------------------------------------------------------
# Import Libraries 
#---------------------------------------------------------------------------------------------

import re
import os
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

#---------------------------------------------------------------------------------------------
# Define Functions
#---------------------------------------------------------------------------------------------

def get_current_version():
    '''
    Returns the current version of the script by getting the last version in the output directory
    '''
    output_dir = "../../Output/"
    # List the folders in the directory
    output_dirs = os.makedirs(output_dir, exist_ok=True)
    lora_output_regex = "^MataLoraAdapter-v(\d+)$"
    version_num = re.search(lora_output_regex, max([d for d in os.listdir(output_dir) if re.match(lora_output_regex, d)], key=lambda x: int(re.search(lora_output_regex, x).group(1))))[1]
    
    return int(version_num)

def main():
    '''
    Main function to train the language model using the MATA framework. This function loads the JSON data from the previous step, loads the model framework, trains it, and saves the updated model.
    '''
    
    # Open the JSON file and load the data 
    path = "../../Input/Clean/"
    file_name = "training_data.jsonl"
    dataset_path = path + file_name

    # Configuration
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    output_dir = "../../Output/MataLoraAdapter-v" + str(get_current_version() + 1)  # Increment version number for new output
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Step 1: Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Step 2: Load the base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        ).to(device)
    
    # Step 4: Configure LoRA
    # This is where the magic happens!
    lora_config = LoraConfig(
        r=16,  # Rank of the update matrices (higher = more parameters, better quality)
        lora_alpha=32,  # Scaling factor (typically 2x the rank)
        target_modules=["q_proj", "v_proj"],  # Which attention layers to adapt
        lora_dropout=0.05,  # Dropout for regularization
        bias="none",  # Don't adapt bias terms
        task_type="CAUSAL_LM"
    )
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    print("Trainable parameters in the model:")
    model.print_trainable_parameters()  # Show how many parameters we're actually training
    
    # Step 5: Load and prepare dataset
    print("Loading dataset...")
    dataset = pd.read_json(dataset_path, lines=True).\
            to_dict(orient = 'records')
    dataset = tokenizer.apply_chat_template(dataset, tokenize = False) # Apply the model's selected chat template to the data
    #dataset = load_dataset_from_jsonl(dataset_path)
    
    return None

    # Step 6: Set training parameters
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,  # Number of complete passes through the data
        per_device_eval_batch_size=1,  # Adjust based on your GPU memory
        gradient_accumulation_steps=4,  # Simulate larger batch size
        gradient_checkpointing = True,
        learning_rate=2e-4,  # Learning rate for LoRA
        fp16=False,  # Use mixed precision training
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",  # Memory-efficient optimizer
        max_length=512,
        dataset_text_field="text",
    )
    
    # Step 7: Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    # Step 8: Start training!
    print("Starting training...")
    trainer.train()
    
    # Step 9: Save the fine-tuned model
    print("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete! Model saved to {output_dir}")
    # Load JSON data from previous step

    # Load model framework


    return(None)

#---------------------------------------------------------------------------------------------
# Run Code
#---------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()

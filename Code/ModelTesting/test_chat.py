# Author: Alden Porter
# This file tests the chat functionality of the unedited model and the new model.

#--------------------------------------------------------------------------
# Import Libraries
#--------------------------------------------------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#--------------------------------------------------------------------------
# Define Globals 
#--------------------------------------------------------------------------

torch.manual_seed(42)
torch.mps.manual_seed(42)

#--------------------------------------------------------------------------
# Define Functions
#--------------------------------------------------------------------------


def main():
    
    test_new = False # Set to True to test the new model
    
    # Select a model, and the hardware to run it on
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    device = device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Define the conversation we want to use as a test
    messages = [
        {"role": "system", "content": "Answer concisely and accurately."},
        {"role": "user", "content": "What is the capital of France?"},
        #{"role": "assistant", "content": "Paris."},
        #{"role": "user", "content": "What country is it in?"},
    ]
    
    print("Loadking Tokenizer")
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading Model")
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    ).to(device)
    
    # Put the model in evaluation mode
    model.eval()

    # Test the chat functionality of the unedited model
    #print("Testing unedited model...")
    
    inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            return_tensors = "pt",
            return_dict = True,
            ).to(device)
    
    # Display formats of inputs
    #print(inputs)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=float(3*10**0),
        )
    
    print("______________________PRINTING ANSWER, ORIGINAL MODEL___________________________")
    answer = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(answer)

   
    if test_new:
        # Test the chat functionality of the new model
        print("Testing new model...")
    # Add code to test the new model here

#--------------------------------------------------------------------------
# Run Code
#--------------------------------------------------------------------------

if __name__ == '__main__':
    main()

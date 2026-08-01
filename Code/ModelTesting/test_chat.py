# Author: Alden Porter
# This file tests the chat functionality of the unedited model and the new model.

#--------------------------------------------------------------------------
# Import Libraries
#--------------------------------------------------------------------------

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
from peft import PeftModel

#--------------------------------------------------------------------------
# Define Globals 
#--------------------------------------------------------------------------

torch.manual_seed(42)
torch.mps.manual_seed(42)

#--------------------------------------------------------------------------
# Define Functions
#--------------------------------------------------------------------------

def save_answer_to_file(answer, filename):
    '''
    This function takes the answer output and saves it to a text file
    '''
    
    with open(filename, "w") as f:
        f.write(answer)

def run_file_in_stata(filename):
    '''
    This function takes a file name and runs it in stata
    Inputs:
        filename, a string, the name of the file to run in stata
    '''
    
    subprocess_result = subprocess.run(['stata-se', 'do', filename], capture_output=True, text=True)
    if subprocess_result.returncode == 0:
        print(subprocess_result.stdout)
    else:
        print(f"Error running Stata file {filename}: {subprocess_result.stderr}")
    
    return None

def extract_stata_code(tokenizer, output, inputs):
    generated_tokens = output[0, inputs["input_ids"].shape[1]:]
    assistant_response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    
    print("Assistant response____________________________________\n", assistant_response, "\n_________________________________________")
    
    return(re.sub('```', '', assistant_response))
    '''
    match = re.search(
        r"```(?:stata|mata)?\s*\n(.*?)```",
        assistant_response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        print("No Stata code block detected")
        return assistant_response  # Return the entire response if no code block is found

    return match.group(1).strip()
    '''
def main():
    
    # Set the path to the LORA Adapter
    version = 1 # Change this to the version of the adapter you want to test
    adapter_path = "../../Output/MataLoraAdapter-v" + str(version)  # Increment version number for new output

    test_new = True # Set to True to test the new model
    sample_bool = False # Set to True to sample from the model, False to use greedy decoding
    temp = float(1*10**0) # Set the temperature for sampling
    
    # Select a model, and the hardware to run it on
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Define the conversation we want to use as a test
    messages = [
        {"role": "system", "content": "Write using the Mata programming language."},
        {"role": "user", "content": "Write a script that prints 'Hello, World!' in Mata and can be run using the stata command. Return only executable code in one ```stata code block; no explanation."},
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
            do_sample=sample_bool,
            temperature=temp,
        )
    
    #old_model_answer = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Old model")
    save_answer_to_file(extract_stata_code(tokenizer, output, inputs), "old_model_test.do")
   
    if test_new:
        # Test the chat functionality of the new model
        print("Loading Model Adapter")
        model = PeftModel.from_pretrained(model, adapter_path).to(device)
        model.eval()
        
        print("Testing new model...")
        with torch.inference_mode():
            output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=sample_bool,
                    temperature=temp)
        
        # Generate asnwer from the new model
        
        print("New model")
        save_answer_to_file(extract_stata_code(tokenizer, output, inputs), "new_model_test.do")

    # Run the two different files in stata
    print("___________________________Running old model test in Stata...")
    run_file_in_stata("old_model_test.do")
    
    print("___________________________Running new model test in Stata...")
    run_file_in_stata("new_model_test.do")

    return()
#--------------------------------------------------------------------------
# Run Code
#--------------------------------------------------------------------------

if __name__ == '__main__':
    main()

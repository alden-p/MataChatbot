# Author: Alden Porter
# This file runs the training, updatining and intercation with the model.

from CleanData.mata_manual_scrape import main as scrape_data
from ModelTraining.mata_train_llm import main as train_model
from ModelTesting.test_chat import main as test_chat

#-------------------------------------------------------------
# Define main
#-------------------------------------------------------------

def main():
    
    # Extract data from the mata manual
    scrape_data()

    # Use the extracated data to train the model
    train_model()

    # Test out the chat for the base versus updated models
    test_chat()

#-------------------------------------------------------------
# Run main
#-------------------------------------------------------------

if __name__ == "__main__":
    main()

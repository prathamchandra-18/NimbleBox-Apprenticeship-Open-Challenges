
# Import the necessary libraries
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
import sys
import numpy as np
import pandas as pd
import os
import fire

# This class defines a dataset for the Netflix data
class NetflixDataset(Dataset):
    @staticmethod
    
    # This function gets the default configuration for the dataset
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    # This function initializes the dataset
    def __init__(self, config, data):
        self.config = config

        # This gets the unique characters in the data.
        self.chars = sorted(list(set(data)))
        
        # This gets the size of the data and the vocabulary.
        data_size, vocab_size = len(data), len(self.chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        # This creates a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        
        # This creates a mapping from integers to characters
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        
        # This stores the vocabulary size
        self.vocab_size = vocab_size
        
        # This stores the data
        self.data = data

    # This gets the vocabulary size
    def get_vocab_size(self):
            return self.vocab_size

    # This gets the block size
    def get_block_size(self):
        return self.config.block_size

    # This gets the number of items in the dataset
    def __len__(self):
        return len(self.data) - self.config.block_size

    # This gets the item at the given index.
    def __getitem__(self, idx):
        
        # This gets a chunk of `block_size + 1` characters from the data.
        chunk = self.data[idx:idx + self.config.block_size + 1]
        
        # This encodes every character to an integer
        dix = [self.stoi[s] for s in chunk]
        
        # This returns the encoded chunk as a tensor
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# This function gets the configuration from the command line, or uses the default values if no arguments are passed.
def get_configration(fp, lr):
    print("file path: ",fp)
    print("learning rate: ",lr)

    C = CN()

    # The `system` section of the configuration contains settings that control the environment, such as the random seed and the working directory.
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = 'out/netflixdata'

    # The `data` section of the configuration contains settings that control the dataset, such as the block size and the vocabulary size.
    C.data = NetflixDataset.get_default_config()

    # The `model` section of the configuration contains settings that control the model, such as the model type and the learning rate
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # The `trainer` section of the configuration contains settings that control the training process, such as the number of epochs and the batch size.
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = lr # the model we're using is so small that we can go a bit faster

    return (C, fp)


# This is the main function
if __name__ == '__main__':

    # Get the configuration and the data path from the command line, or use the default values if no arguments are passed
    (config, data_path) = fire.Fire(get_configration)
    
    # Set the random seed
    set_seed(config.system.seed)

    # Load the training data from the CSV file.
    training_data_df = pd.read_csv(data_path)
    training_data_text = training_data_df['description'].str.cat(sep="\n")    
    
    # Create a dataset object for the training data.
    train_dataset = NetflixDataset(config.data, training_data_text)

    # Update the configuration with the block size and vocabulary size from the dataset
    config.model.block_size = train_dataset.get_block_size()
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.block_size = train_dataset.get_block_size()
    config.vocab_size = train_dataset.get_vocab_size()
    config.chars = train_dataset.chars
    
    # Set up logging.
    setup_logging(config)

    # Instantiate the model
    model = GPT(config.model)
    
    # Load the model state from the checkpoint file.
    model.load_state_dict(torch.load('out/netflixdata/model.pt', torch.device('cpu')))

    # Print the configuration of the model
    print(config)

    # Create a trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # Define a callback function that prints the iteration time, iteration number, and training loss every 10 iterations
    def callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # Evaluate the model on the training and test sets
            model.eval()
            # Save the model state to a checkpoint file
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # Restore the model to training mode
            model.train()

    # Set the callback function for the trainer
    trainer.set_callback('on_batch_end', callback)

    # Run the optimization
    trainer.run()


# Import the necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer
import pandas as pd
import json

# Define the GenerateRequest class, which will be used to pass in the input text and maximum length to the generate_text function
class GenerateRequest(BaseModel):
    text: str
    max_length: int

# Load the configuration data from the config.json file
with open('out/netflixdata/config.json','r') as f:
    data = json.load(f)

# Create a FastAPI app
app = FastAPI()

# Create a GPT model and load the model weights from the model.pt file
config = GPT.get_default_config()
config.vocab_size = data['vocab_size']
config.block_size = data['block_size']
config.model_type = 'gpt-mini'
model = GPT(config)
chars = data['chars']


model.load_state_dict(torch.load("out/netflixdata/model.pt",map_location=torch.device('cpu')))
device = 'cpu'

@app.post("/generatetext")
def generate_text(request: GenerateRequest):
# def generate_text(text, max_length):
    
    # Create a dictionary that maps characters to their indices in the vocabulary
    stoi = { ch:i for i,ch in enumerate(chars) }
    
    # Create a dictionary that maps indices in the vocabulary to characters
    itos = { i:ch for i,ch in enumerate(chars) }

    # Get the input text and maximum length from the request
    input_text = request.text
    max_length = request.max_length
    
    # If the input_text is empty, set it to a single space character
    if(input_text == ''): input_text = ' '
    
    # Convert the input_text to a tensor of indices in the vocabulary
    x = torch.tensor([stoi[s] for s in input_text], dtype=torch.long)[None,...]
    
    # Generate text using the GPT model
    y = model.generate(x, max_length, temperature=1.0, do_sample=True, top_k=10)[0]
    
    # Convert the generated text back to a string
    completion = ''.join([itos[int(i)] for i in y])

    # Return the generated text
    return {"generated_text":completion}

        



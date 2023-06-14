# NimbleBox-Apprenticeship-Open-Challenges
This repository contains the challenge by NimbleBox.ai for the ML Engineer internship role.<br/>

## Instructions
To get started, download all the files by running the following command locally: `git clone https://github.com/prathamchandra-18/NimbleBox-Apprenticeship-Open-Challenges.git`<br/>
Next, initialize the submodule using the command: `git submodule init` <br/>
To update the submodules from the remote, run: `git submodule update --force --recursive --init --remote`<br/>
Imprt the minGPT directory by the following command: `git clone https://github.com/karpathy/minGPT.git`<br/>
Change the directory to `minGPT/` and install it by running the command: <br/>
`pip install -e .` <br/>
Now we are ready to proceed with training and serving our model.

### Training
To train the model further using the dataset, use the following command: <br/>
`python trainer.py --fp 'netflix_titles.csv' --lr 0.001` <br/>
This command will initiate training from the last checkpoint and save the model weights and configurations into the `out/netflixdata` directory. <br/>

### Serving
To serve the model, run the server using the following command: <br/>
`python -m uvicorn server:app --reload` <br/>
This will start a server locally with the IP address 127.0.0.1 and port number 8000, and the endpoint will be '/generatetext'.<br/>
To send a request to this server, use the following JSON format: `{"text": "Enter Your Prompt Here", "max_length": "Enter the maximum length of your reply here"}`<br/>
The output message will be in the format: `{"generated_text":"This is the generated text."}`<br/>

To use these commands for CURL (Windows): <br/>
`set json={"text": "NimbleBox!", "max_length": 1000}` <br/>
`curl -i -X POST -H "Content-Type: application/json" -d "%json:"=\"%" http://127.0.0.1:8000/generatetext` <br/>

### Multithreading
-Stress Testing the server: To perform stress testing, execute the following command: <br/> `python test.py --url http://127.0.0.1:8000/generatetext --threads 10 --requests 10 --max_length 500`<br/>
-Multithreaded Inference: To perform multithreaded inference from our model, utilize this command: <br/> `python test.py --url http://127.0.0.1:8000/generatetext --messages "This" "is" "my" "Assignment" "for" "NimbleBox" "Internship" --max_length 500`








# Import necessary modules
import requests
import threading
import argparse
import time

# Define a function to perform a stress test on the server
def stress_test(url, num_threads, num_requests, max_length):
    
    # Create a function to make requests to the server
    def make_requests():
        for _ in range(num_requests):
            # Send a request to the server
            payload = {"text": '', "max_length": max_length}
            response = requests.post(url, json=payload)
            # Print the response status code
            print(f"Response status code: {response.status_code}")
    
    # Create a list of threads
    threads = []
    
    # Create a thread for each request
    for _ in range(num_threads):
        # Create a thread that sends requests to the server
        thread = threading.Thread(target=make_requests)
        # Add the thread to the list of threads
        threads.append(thread)
        # Start all of the threads
        thread.start()

    # Wait for all of the threads to finish
    for thread in threads:
        # Join the thread
        thread.join()

# Define a function to perform fast inference using multiple threads
def fast_inference(url, messages, max_length):
    
    # Create a function to send a message to the server for inference
    def send_message(message):
        # Send a request to the server for inference
        payload = {"text": message, "max_length": max_length}
        response = requests.post(url, json=payload)
        # Print the response status code and the response
        print(f"Response status code: {response.status_code}")
        print(response.json())
    
    # Create a list of threads
    threads = []
    
    # Create a thread for each message
    for i in range(len(messages)):
        # Create a thread that sends a message to the server for inference
        thread = threading.Thread(target=send_message, args=(messages[i],))
        # Add the thread to the list of threads
        threads.append(thread)
        # Start the thread
        thread.start()

    # Wait for all of the threads to finish
    for thread in threads:
        # Join the thread
        thread.join()

# Define the main function    
def main():
    
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Server Stress Test and Model Fast Inference CLI')
    # Add arguments to the parser
    parser.add_argument('--url', type=str, help='URL of the server')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads to use for stress testing')
    parser.add_argument('--requests', type=int, default=1, help='Number of requests per thread for stress testing')
    parser.add_argument('--max_length', type=int, help='Maximum number of characters to be output by the model')
    parser.add_argument('--messages', nargs="*", type=str, default=[], help='All the prompts')
    # Parse the arguments
    args = parser.parse_args()

    # Check if the required arguments are provided
    if args.url and args.threads > 0 and args.requests > 0:
        # Perform stress test
        stress_test(args.url, args.threads, args.requests, args.max_length)
    elif args.url and args.messages:
        # Perform fast inference
        fast_inference(args.url, args.messages, args.max_length)
    else:
        # Print an error message
        print("Not valid arguments!")


# Check if the script is being run as the main program
if __name__ == "__main__":
    main()

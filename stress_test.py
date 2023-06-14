"""
This program performs a stress test on a server.

It sends a specified number of requests to the server, using a specified number of threads.

The program prints the status code of each response.
"""

import requests
import threading
import argparse
import time

def stress_test(url, num_threads, num_requests, max_length):
    """
    Performs a stress test on the server.

    Args:
        url: The URL of the server.
        num_threads: The number of threads to use for the stress test.
        num_requests: The number of requests per thread for the stress test.
        max_length: The maximum length of a message.
    """
    
    def make_requests():
        for _ in range(num_requests):
            payload = {"text":'',"max_length":max_length}
            response = requests.post(url, json=payload)
            print(f"Response status code: {response.status_code}")
    # Create a list of threads
    threads = []
    for _ in range(num_threads):
        # Create a thread that sends requests to the server
        thread = threading.Thread(target=make_requests)
        # Add the thread to the list of threads
        threads.append(thread)
        # Start all of the threads.
        thread.start()

    # Wait for all of the threads to finish
    for thread in threads:
        thread.join()


def main():
    """
    The main function.
    """

    # Create an argument parser.
    parser = argparse.ArgumentParser(description='Server Stress Test, Messaging, and Model CLI')
    
    # Add arguments to the parser
    parser.add_argument('--url', type=str, help='URL of the server')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads to use for stress testing')
    parser.add_argument('--requests', type=int, default=1, help='Number of requests per thread for stress testing')
    parser.add_argument('--max_length',type=int, help='Maximum number of characters to be outputed by the model')
    
    # Parse the arguments
    args = parser.parse_args()

    # Check that the arguments are valid
    if args.url and args.threads > 0 and args.requests > 0:
        # Perform a stress test on the server
        stress_test(args.url, args.threads, args.requests, args.max_length)
    
    else:
        # Print an error message
        print("Not valid arguments!")



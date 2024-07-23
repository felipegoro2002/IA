import requests
import uuid
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# URL of the API endpoint
url = "https://nn1etsjuy11z8t-4000.proxy.runpod.net/generate"

# Headers required for the API request
headers = {
    "Content-Type": "application/json",
    "Cookie": "__cflb=02DiuGz3huhhVux26zX9GPq8X3AXuduC2kBJ4ccQLh4aG; __cflb=02DiuGz3huhhVux26zVY5tw5fAQEvh17kK1wtQt78gpJ8"
}

# List of food items to test with
food_items = ["BARBACUE", "PIZZA", "SUSHI", "BURGER", "PASTA"]

# Number of total tests to perform
total_tests = 50

def make_request(food_item):
    # Generate a unique UUID
    unique_id = str(uuid.uuid4())
    
    # Create the payload for the POST request
    data = {
        "prompt": food_item,
        "uuid": unique_id
    }
    
    # Send the POST request
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    # Print the response from the server
    print(f"Response for {food_item} with UUID {unique_id}:")
    print(response.text)
    print("\n")
    return response

# Create a ThreadPoolExecutor to run requests concurrently
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    for _ in range(total_tests):
        food_item = random.choice(food_items)
        futures.append(executor.submit(make_request, food_item))
    
    # Wait for all futures to complete
    for future in as_completed(futures):
        future.result()

import requests
import json
import time
from tabulate import tabulate

# Text to classify
text = "You have won a free trip to the Bahamas! Claim your prize now!"

# Username
username = "oussama"

# URL of your FastAPI endpoint
url = "http://localhost:8000/classify/?text={}&username={}".format(text, username)

try:
    # Make the POST request and time it
    time_start = time.time()
    response = requests.post(url)
    time_end = time.time()

    # Calculate response time
    response_time = int((time_end - time_start) * 1000)  # in milliseconds

    # Check if the response is successful
    if response.status_code == 200:
        # Parse JSON response
        result = json.loads(response.content.decode('utf-8'))

        # Extract the label and probability
        label = result["label"]
        probability = result["probability"]

        # Create a table to display the results
        table = [["Text", "Label", "Probability", "Response Time (ms)", "Source"],
                 [text, label, probability, response_time, result["source"]]]

        # Print the table
        print(tabulate(table, headers="firstrow", tablefmt="grid"))

    else:
        print("Failed to receive a successful response from the server")
        print("Status code:", response.status_code)
        print("Response text:", response.text)

except Exception as e:
    print("An error occurred:", e)

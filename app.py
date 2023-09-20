from flask import Flask, jsonify
import csv
import random
import time
import json

app = Flask(__name__)

# Function to create a CSV file with header
def create_csv(filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Flag1', 'Flag2', 'Flag3', 'Flag4', 'Flag5']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header row
        writer.writeheader()

# Function to update the CSV file with new flag values
def update_csv(filename, new_flags):
    with open(filename, 'a', newline='') as csvfile:  # Use 'a' for append mode
        fieldnames = ['Flag1', 'Flag2', 'Flag3', 'Flag4', 'Flag5']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write new flag values to a new row
        writer.writerow(new_flags)

# Create the CSV file with header if it doesn't exist
create_csv('flags.csv')

@app.route("/")
def index():
    return "Welcome to the Flags App!"

@app.route("/update")
def update_flags():
    # Generate random True/False values for the flags
    new_flags = {
        'Flag1': random.choice([True, False]),
        'Flag2': random.choice([True, False]),
        'Flag3': random.choice([True, False]),
        'Flag4': random.choice([True, False]),
        'Flag5': random.choice([True, False])
    }
    
    # Update the CSV file with the new flag values
    update_csv('flags.csv', new_flags)
    
    return "Flags updated successfully."

@app.route("/csv_to_json")
def csv_to_json():
    data = []
    
    # Read the CSV file and convert it to a list of dictionaries
    with open('flags.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    
    # Convert the data to JSON format
    json_data = json.dumps(data)
    
    return json_data

if __name__ == "__main__":
    app.run(debug=True)

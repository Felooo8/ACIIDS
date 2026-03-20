import json
import os, sys
sys.path.insert(0, os.path.abspath("...."))
from config import NEWS_CLEANED_FILE, NEWS_DIR

# Function to remove the sentiment field from the JSON data
def remove_sentiment_field(json_path, output_path):
    # Load the original JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Iterate through the data to remove the 'sentiment' field
    for article in data:
        if 'sentiment' in article:
            del article['sentiment']

    # Save the new JSON file without the sentiment field
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Updated JSON saved to {output_path}")

# Example usage
input_file = NEWS_DIR
output_file = NEWS_CLEANED_FILE
remove_sentiment_field(input_file, output_file)

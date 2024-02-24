import torch
import requests
import re
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

def postprocess_response(outputs, tokenizer):
    response = tokenizer.decode(outputs.start_logits, skip_special_tokens=True)
    return response

def fetch_nutrition_info(query):
    # Replace with your actual API key and other necessary information
    api_key = "53cb29fcbbd2fcbaa9d1a9b0670a0fda"
    app_id = "20006eeb"
    base_url = "https://api.edamam.com/api/nutrition-data"

    # Prepare the request parameters
    params = {
        "app_id": app_id,
        "app_key": api_key,
        "q": query,
    }

    # Send the request and get the response
    response = requests.get(base_url, params=params)
    data = response.json()

    # Extract the nutritional information
    nutrition_info = data["hints"][0]["food"]["nutrients"]

    # Format the response
    response_text = "Here's the nutritional information for that food:\n"
    for nutrient, value in nutrition_info.items():
        response_text += f"{nutrient}: {value} {value['unit']}\n"

    return response_text

def handle_user_query(query):
    # Preprocess the query
    inputs = tokenizer(query, return_tensors="pt")

    # Generate a response
    outputs = model(**inputs)

    # Post-process the outputs
    response = tokenizer.decode(outputs.start_logits, skip_special_tokens=True)

    return response

def main():
    print("Welcome to the Fitness Chatbot!")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "quit":
            break

        response = handle_user_query(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
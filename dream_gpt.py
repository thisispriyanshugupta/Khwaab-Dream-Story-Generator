import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the saved model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./saved_model")
tokenizer = GPT2Tokenizer.from_pretrained("./saved_model")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model to evaluation mode
model.eval()

# Function to generate dream story
def generate_dream_story(dream_description, max_length):
    # Tokenize the dream description
    tokenized_prompt = tokenizer(dream_description, return_tensors="pt")

    # Move input tensors to the appropriate device
    input_ids = tokenized_prompt["input_ids"].to(device)
    attention_mask = tokenized_prompt["attention_mask"].to(device)

    # Generate text using the model
    output = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=max_length,
                            num_return_sequences=1,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id,
                            do_sample=True,  # Enable sampling
                            top_k=50,         # Adjust top-k sampling parameter
                            top_p=0.95)       # Adjust nucleus sampling parameter

    # Decode the generated output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Ensure the generated text ends with a complete sentence
    sentences = generated_text.split(".")
    if len(sentences) > 1:
        generated_text = ".".join(sentences[:-1]) + "."

    return generated_text

# Main function for user interaction
def main():
    # Prompt the user to input the dream description
    dream_description = input("Enter the dream description: ")

    # Prompt the user to input the desired maximum length
    max_length_input = input("Enter the maximum length for the generated text: ")
    max_length = int(max_length_input)

    # Generate dream story
    generated_text = generate_dream_story(dream_description, max_length)

    # Print the generated text
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()

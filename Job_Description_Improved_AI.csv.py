import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
import pandas as pd

# Load the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define device for GPU performance rtx goes brrrrrr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def predict_education_level(input_text, model):
    # Tokenize input text, use embedding ??
    inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get the ouput
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Find index of the label
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    # Map predicted class to education level label
    education_levels = ["PHD degree", "bachelor's degree", "master's degree"]
    predicted_level = education_levels[predicted_class]

    return predicted_level


# Load saved model if exists
if (os.path.isfile('saved_model.pth')):
    model.load_state_dict(torch.load('saved_model.pth'))
    print("Model loaded successfully!")
    while True:
        user_input = input("Enter job description (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            exit(0)
        else:
            predicted_level = predict_education_level(user_input, model)
            print("Predicted education level:", predicted_level)

# Sample data
df = pd.read_csv("dataset/updated_job_descriptions.csv")

# Extract job descriptions and qualifications
job_descriptions = df["Job Description"].tolist()
labels = df["Qualifications"].tolist()

# Reduce the number of data --> takes too much time
max_length = 1024
job_descriptions = job_descriptions[:max_length]
labels = labels[:max_length]

tokenized_inputs = tokenizer(job_descriptions, padding=True, truncation=True, return_tensors='pt')

# Extract input_ids and attention_masks from tokenized_inputs
input_ids = tokenized_inputs['input_ids']
attention_masks = tokenized_inputs['attention_mask'] # Make sur AI doesn't check padding that berts adds
labels = torch.tensor(labels)

# Create train and validation sets
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, test_size=0.2,
                                                                                    random_state=42)
# Define batch size for training
batch_size = 8

# Create TensorDatasets
train_dataset = TensorDataset(train_inputs, attention_masks[:len(train_inputs)], train_labels)
validation_dataset = TensorDataset(validation_inputs, attention_masks[len(train_inputs):], validation_labels)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

# Fine-tuning parameters
epochs = 50
learning_rate = 2e-5

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epochs)
total_steps = len(train_dataloader) * epochs

print("Starting training")

# Fine-tuning the model
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Perform gradient accumulation

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)

    print(f'Epoch [{epoch + 1}/{epochs}], '
          f'Average Training Loss: {avg_train_loss:.4f}')

    # Print accuracy here ?
    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch in validation_dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_eval_loss += loss.item()

# Save the model
torch.save(model.state_dict(), 'saved_model.pth')

# User interaction loop
while True:
    user_input = input("Enter job description (type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    else:
        predicted_level = predict_education_level(user_input, model)
        print("Predicted education level:", predicted_level)

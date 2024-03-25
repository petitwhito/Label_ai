import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup

# Load the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Sample data (needs upgrading)
job_descriptions = [
    "This job requires a Bachelor's degree in Computer Science or related field.",
    "Candidates should hold a Master's degree in Finance or equivalent qualification.",
    "No formal education is required for this entry-level position, but relevant experience is preferred.",
    "Bachelor's degree in Engineering preferred.",
    "Minimum requirement: Master's degree in Economics.",
    "No school diploma or equivalent required.",
    "Seeking candidates with a PhD in Mathematics.",
    "Experience in the field is preferred but not required.",
    "Degree in Master Business Administration or related field is a plus.",
    "Bachelor's degree in cybersecurity or related field is mandatory.",
    "Applicants must have at least a bachelor's degree in Marketing.",
    "Doctorate degree in Physics or a related discipline is highly desirable.",
    "Applicants must possess a graduate degree or master's in Mechanical Engineering or a closely related discipline.",
    "We are seeking candidates with a postgraduate qualification in Data Science or a relevant field.",
    "No educational background is necessary for this role; we value practical skills and hands-on experience.",
    "Candidates are not required to have any academic qualifications; a keen interest in the subject matter is "
    "preferred.",
]

# Labels
labels = [1, 2, 0, 1, 2, 0, 1, 0, 2, 1, 1, 1, 2, 2, 0, 0]
# 0 for no formal education, 1 for bachelor's
# degree, 2 for master's degree

tokenized_inputs = tokenizer(job_descriptions, padding=True, truncation=True, return_tensors='pt')

# Extract input_ids and attention_masks from tokenized_inputs
input_ids = tokenized_inputs['input_ids']
attention_masks = tokenized_inputs['attention_mask']
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
epochs = 60
learning_rate = 2e-5

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epochs)
total_steps = len(train_dataloader) * epochs

# Define device
device = torch.device('cpu') # Yeah i didn't know gpu existed
model.to(device)

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

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    # Calculate average loss
    avg_train_loss = total_loss / len(train_dataloader)
    # Print training and validation metrics
    print(f'Epoch [{epoch + 1}/{epochs}], '
          f'Average Training Loss: {avg_train_loss:.4f}')

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


def predict_education_level(input_text):
    model.eval()
    # Tokenize input text
    inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        print(outputs.logits)

    # Get predicted class (education level)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    # Map predicted class to education level label
    education_levels = ["no formal education", "bachelor's degree", "master's degree"]
    predicted_level = education_levels[predicted_class]

    return predicted_level

# FIXME: model saving

# User interaction loop
while True:
    user_input = input("Enter job description (type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    else:
        predicted_level = predict_education_level(user_input)
        print("Predicted education level:", predicted_level)

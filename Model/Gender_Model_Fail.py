import torch
import torch.nn as nn
import torch.optim as optim
import torchtext.vocab as vocab
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

# Prepare Data
data = pd.read_csv('../dataset/name_gender_dataset.csv')
names = data['Name'].tolist()
labels = data['Gender'].tolist()

# Define tokenizer
tokenizer = get_tokenizer('basic_english')

# Tokenize names
names_tokenized = [tokenizer(name) for name in names]

# Load pre-trained GloVe model
glove = vocab.GloVe(name='6B', dim=100)

# Convert tokens to indices using GloVe vocab, filtering out tokens not in the vocabulary ???
name_indices = [[glove.stoi[token] for token in tokens if token in glove.stoi] for tokens in names_tokenized]
name_indices = [indices for indices in name_indices if indices]
name_indices = pad_sequence([torch.tensor(indices) for indices in name_indices], batch_first=True, padding_value=0)


# Define Model Architecture
class GenderClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(GenderClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(glove.vectors)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)  # Average pooling
        hidden = torch.relu(self.fc(pooled))
        output = self.fc2(hidden)
        return output


# Train the Model
input_dim = 100
hidden_dim = 64
output_dim = 1

model = GenderClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert string labels to numerical values
filtered_labels_numeric = [1.0 if label == 'F' else 0.0 for label, indices in zip(labels, name_indices) if
                           len(indices) > 0]

# Create tensor from numerical labels
filtered_labels = torch.tensor(filtered_labels_numeric, dtype=torch.float32)

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(name_indices)
    loss = criterion(outputs.squeeze(), filtered_labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')


def predict_gender(name):
    tokens = tokenizer(name)
    tensor = torch.tensor([glove.stoi[token] for token in tokens if token in glove.stoi])
    tensor = tensor.unsqueeze(0)  # Add a batch dimension
    predictions = torch.sigmoid(model(tensor))
    gender = 'male' if predictions.item() >= 0.5 else 'female'
    return gender

# Doesn't work :(
def calculate_precision(model, names_indices, labels):
    male_TP = 0
    male_FP = 0
    female_TP = 0
    female_FP = 0

    # Get the model's raw predictions
    with torch.no_grad():
        outputs = model(names_indices)

    # Convert raw predictions to binary predictions (0 or 1)
    binary_predictions = (torch.sigmoid(outputs) >= 0.5).squeeze().int()

    # Iterate through the binary predictions and true labels
    for prediction, label in zip(binary_predictions, labels):
        # Increment the appropriate counters based on the model's prediction and the true label
        if label == 0 and prediction == 1:
            male_FP += 1
        elif label == 1 and prediction == 0:
            female_FP += 1
        elif label == 1 and prediction == 1:
            female_TP += 1
        elif label == 0 and prediction == 0:
            male_TP += 1

    # Calculate precision for male and female genders
    male_precision = male_TP / (male_TP + male_FP) if (male_TP + male_FP) > 0 else 0
    female_precision = female_TP / (female_TP + female_FP) if (female_TP + female_FP) > 0 else 0

    return male_precision, female_precision


while True:
    name = input("Enter your name: ")
    res = predict_gender(name)
    print("Your gender is: " + res)

import fasttext.util
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

# Download and load the pre-trained FastText model
fasttext.util.download_model('en', if_exists='ignore')  # Download English model
ft = fasttext.load_model('cc.en.300.bin')  # Load the downloaded model

# Extract data from CSV file
data = pd.read_csv("../dataset/name_gender_dataset.csv")
data['Name'] = data['Name'].apply(lambda x: x.strip().lower())
data['Gender'] = data['Gender'].apply(lambda x: 0 if x.strip() == 'M' else 1)

X = data['Name']
Y = data['Gender']


# Feature engineering: Add number of vowels
def count_vowels(name):
    vowels = 'aeiou'
    return sum(1 for char in name if char in vowels)


data['NumVowels'] = data['Name'].apply(count_vowels)

# Split the data into training and test
X = data[['Name', 'NumVowels']]  # Include the number of vowels as a feature
Y = data['Gender']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Extract features using pre-trained word vectors
def extract_word_vectors(names):
    vectors = []
    for name in names:
        name_vec = ft.get_sentence_vector(name[0])  # Get word vector for the name
        vectors.append(np.concatenate([name_vec, [name[1]]]))  # Include NumVowels as feature
    return torch.tensor(vectors, dtype=torch.float)


X_train_vec = extract_word_vectors(X_train.values)
X_test_vec = extract_word_vectors(X_test.values)


# Create the model with PyTorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(301, 128)  # Adjusted input size to include NumVowels ? Probably not needed
        self.dropout = nn.Dropout(0.5)  # Dropout layer for regularization ?
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


net = Net()

# Preparing training data
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
X_train_tensor = X_train_vec
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float).view(-1, 1)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    y_pred = net(X_train_tensor)
    loss = criterion(y_pred, Y_train_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# Preparing testing data
X_test_tensor = X_test_vec
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float).view(-1, 1)

# Testing the model
with torch.no_grad():
    y_pred = net(X_test_tensor)
    predicted = torch.round(y_pred)
    accuracy = torch.sum(predicted == Y_test_tensor) / len(predicted)
    print('Test Accuracy: {:.4f}'.format(accuracy.item()))


# Prediction function
def predict_gender(name, num_vowels):
    name_vec = ft.get_sentence_vector(name)
    input_vec = torch.tensor(np.concatenate([name_vec, [num_vowels]]), dtype=torch.float).unsqueeze(0)
    with torch.no_grad():
        y_pred = net(input_vec)
        gender = 'Male' if y_pred.item() < 0.5 else 'Female'
    return gender


# Interaction loop
while True:
    name = input("Enter your name: ")
    num_vowels = count_vowels(name)
    res = predict_gender(name, num_vowels)
    print("Predicted gender: " + res)

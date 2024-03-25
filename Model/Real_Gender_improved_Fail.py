import fasttext.util
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np

# Tried to improve performance with fasttext but failed !

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
    vowels = 'aeiouAEIOU'
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
        vectors.append(np.concatenate([name_vec, [name[1]]]))  # Include NumVowels as feature ?? Probably bad idea
    return torch.tensor(vectors, dtype=torch.float)

X_train_vec = extract_word_vectors(X_train.values)
X_test_vec = extract_word_vectors(X_test.values)

# Create the model with PyTorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(301, 256)  # Increase number of neurons in the first layer
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)  # Add another hidden layer
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)  # Another one for even more precision
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x): # Foward function
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
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
num_epochs = 20
batch_size = 64
for epoch in range(num_epochs):
    X_train_tensor, Y_train_tensor = shuffle(X_train_tensor, Y_train_tensor)  # Shuffle training data
    for i in range(0, len(X_train_tensor), batch_size):
        optimizer.zero_grad()
        batch_X, batch_Y = X_train_tensor[i:i+batch_size], Y_train_tensor[i:i+batch_size]
        outputs = net(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
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
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Extract data from CSV file

data = pd.read_csv("dataset/name_gender_dataset.csv")

data['Name'] = data['Name'].apply(lambda x: x.strip()).str.lower()
data['Gender'] = data['Gender'].apply(lambda x: x.strip())
data['Gender'] = data['Gender'].apply(lambda x: 0 if x == 'M' else 1)

X = data['Name']
Y = data['Gender']

# Split the data into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Transform the data into numerical vector for nn
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 2))  # Modification needed ?
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Create the model with pytorch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_train_vec.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


net = Net()

# Preparing training data
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
X_train_tensor = torch.Tensor(X_train_vec.toarray())
Y_train_tensor = torch.Tensor(Y_train.values).view(-1, 1)

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
        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))

# preparing testing data

X_test_tensor = torch.Tensor(X_test_vec.toarray())
Y_test_tensor = torch.Tensor(Y_test.values).view(-1, 1)

# Testing the model

with torch.no_grad():
    y_pred = net(X_test_tensor)
    predicted = torch.round(y_pred)
    accuracy = torch.sum(predicted == Y_test_tensor) / len(predicted)
    print('Test Accuracy: {:.4f}'.format(accuracy.item()))


def predict_gender(name):
    name = name.strip().lower()
    name_vec = vectorizer.transform([name])
    name_vec = torch.Tensor(name_vec.toarray())
    with torch.no_grad():
        y_pred = net(name_vec)
        gender = 1 if y_pred.item() > 0.5 else 0
    if gender == 0:
        gender = 'Male'
    else:
        gender = 'Female'
    return gender


while True:
    name = input("Enter your name: ");
    res = predict_gender(name)
    print("Your gender is: " + res)

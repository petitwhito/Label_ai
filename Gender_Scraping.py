import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


# Web Scraping function

def web_scraping(home_url, list_strings):
    all_name = []
    all_gender = []
    for j in range(len(list_strings)):
        response = requests.get(home_url)
        if response.status_code != 200:
            continue
        soup_home = BeautifulSoup(response.content, 'html.parser')
        english_names_link = soup_home.find('a', href='/names/usage/' + list_strings[j])
        if not english_names_link:
            continue
        english_names_url = urljoin(home_url, english_names_link['href'])
        response_english_names = requests.get(english_names_url)
        if response_english_names.status_code != 200:
            continue
        soup_english_names = BeautifulSoup(response_english_names.content, 'html.parser')

        name_elements = soup_english_names.find_all('a', class_='nll')
        names = [name.text.strip() for name in name_elements]
        genders_elements = soup_english_names.find_all('span', class_=lambda value: value and value in ['masc', 'fem'])
        genders = [gender.text.strip() for gender in genders_elements]

        all_name.extend(names)
        all_gender.extend(genders)

        for i in range(2, 16):
            home_url = english_names_url
            response = requests.get(home_url)
            if response.status_code != 200:
                continue
            soup_home = BeautifulSoup(response.content, 'html.parser')
            english_names_link = soup_home.find('a', href='/names/usage/' + list_strings[j] + '/' + str(i))
            if not english_names_link:
                continue
            english_names_url = urljoin(home_url, english_names_link['href'])
            response_english_names = requests.get(english_names_url)
            if response_english_names.status_code != 200:
                continue
            soup_english_names = BeautifulSoup(response_english_names.content, 'html.parser')

            name_elements = soup_english_names.find_all('a', class_='nll')
            names = [name.text.strip() for name in name_elements]
            genders_elements = soup_english_names.find_all('span',
                                                           class_=lambda value: value and value in ['masc', 'fem'])
            genders = [gender.text.strip() for gender in genders_elements]
            all_name.extend(names)
            all_gender.extend(genders)

    return all_name, all_gender


# Function to create database if not already exists
def make_database(home_url, list_country):
    csv_file_path = 'dataset/names_and_genders.csv'
    if os.path.exists(csv_file_path):
        return
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Gender'])
        all_name, all_gender = web_scraping(home_url, list_country)
        data = list(zip(all_name, all_gender))
        writer.writerows(data)
        file.close()
    file.close()

    print(f'Data saved to {csv_file_path}')


url = 'https://www.behindthename.com/'
list_cr = ["english", "french", "german", "italian", "spanish", "arabic", "indian", "irish", "chinese", "japanese",
           "korean", "vietnamese", "african", "dutch", "polish", "russian", "swedish"]

#list_cr = ["english", "french"]

make_database(url, list_cr)

# Reading the csv file
data = pd.read_csv('dataset/names_and_genders.csv')

#check if the data has been opened correctly

data['Name'] = data['Name'].apply(lambda x: x.strip())
data['Gender'] = data['Gender'].apply(lambda x: x.strip())

# Feature Engineering

data['name_length'] = data['Name'].apply(lambda x: len(x))
data['vowel_count'] = data['Name'].apply(lambda x: sum(1 for char in x if char.lower() in 'aeiou'))

X = data[['name_length', 'vowel_count']]

data['Gender'] = data['Gender'].apply(lambda x: x.strip())
data['Gender'] = data['Gender'].apply(lambda x: 0 if x == 'm' else 1)
Y = data['Gender']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train_tensor = torch.Tensor(X_train.values)
X_test_tensor = torch.Tensor(X_test.values)
Y_train_tensor = torch.Tensor(Y_train.values).view(-1, 1)
Y_test_tensor = torch.Tensor(Y_test.values).view(-1, 1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128)
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
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    output = net(X_train_tensor)
    loss = criterion(output, Y_train_tensor)


    l2_regularization = 0
    for param in net.parameters():
        l2_regularization += torch.norm(param, 2) ** 2
    loss += l2_regularization * 0.01

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch + 1:02} | Loss: {loss.item():.4f}')

# Evaluation the model
with torch.no_grad():
    output = net(X_test_tensor)
    predicted = torch.round(output)
    print(f'Accuracy: {accuracy_score(Y_test_tensor, predicted)}')


def predict_gender(name):
    name_len = len(name)
    vowel_count = sum(1 for char in name if char.lower() in 'aeiouAEIOU')

    features = torch.Tensor([[name_len, vowel_count]])

    with torch.no_grad():
        output = net(features)
        proba = output.item()

        # Convert probability to gender label
    return 'male' if proba >= 0.5 else 'female'


while True:
    name = input('Enter your name: ')
    gender = predict_gender(name)
    print(f'{name} is a {gender}')

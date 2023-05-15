#################################################
#                                               #
#   Team Tidal                                  #
#   Kelvin Chen, Daniel Laszczych,              #
#   Stanley Wong, Soroush Semerkant             #
#                                               #
#   This is the script for our Predictive Model #
#   using PyTorch run on Google Colab           # 
#                                               #
#################################################

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Read and shuffle our data

data_df = pd.read_csv("data.csv")
data_df = data_df.sample(frac=1, random_state=42)
data_df.head()

# Get an idea of how much data we have for each coral density level

sns.countplot(x = 'Coverage', data=data_df)

# Convert our output into appropriate labels for our model

coverageToLabels = {
    '1L': 0,
    '1U': 1,
    '2L': 2,
    '2U': 3,
    '3L': 4,
    '3U': 5,
    '4L': 6,
    '4U': 7,
    '5L': 7,
    '5U': 7,
}
labelsToCoverage = {v: k for k, v in dict(list(coverageToLabels.items())[:8]).items()}

data_df['Coverage'].replace(coverageToLabels, inplace=True)
data_df['Prev Coverage'].replace(coverageToLabels, inplace=True)

# Get our input features

x = data_df.iloc[:, list(range(3, 3)) + list(range(5, data_df.shape[1]))]
x.head()

# Get our output features

y = data_df.iloc[:, 3]
y.head()

# Split our data into train, val, and test. Use stratify to distribute our data evenly

x_trainval, x_test, y_trainval, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=21)
x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=42)

# Normalize our inputs so that they are between (0, 1)

scale = MinMaxScaler()
x_train = scale.fit_transform(x_train)
x_val = scale.transform(x_val)
x_test = scale.transform(x_test)
x_train, y_train = np.array(x_train), np.array(y_train)
x_val, y_val = np.array(x_val), np.array(y_val)
x_test, y_test = np.array(x_test), np.array(y_test)

# Get the count of each output class

def getClassDistribution(obj):
    count_dict = {
        '1L': 0,
        '1U': 0,
        '2L': 0,
        '2U': 0,
        '3L': 0,
        '3U': 0,
        '4L': 0,
        '4U': 0,
    }
    
    for i in obj:
        if i == 0: 
            count_dict['1L'] += 1
        elif i == 1: 
            count_dict['1U'] += 1
        elif i == 2: 
            count_dict['2L'] += 1
        elif i == 3: 
            count_dict['2U'] += 1
        elif i == 4: 
            count_dict['3L'] += 1
        elif i == 5: 
            count_dict['3U'] += 1
        elif i == 6: 
            count_dict['4L'] += 1
        elif i == 7: 
            count_dict['4U'] += 1     
            
    return count_dict

# Create a data loader for our model

class createDataset(Dataset):   
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

train_dataset = createDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
val_dataset = createDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())
test_dataset = createDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())

# Create weights to account for the discrepancy between the number of data we have for some coral density levels

target_list = []
for _, t in train_dataset:
    target_list.append(t)
    
target_list = torch.tensor(target_list)

class_count = [i for i in getClassDistribution(y_train).values()]
class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
print(f'Class Weights: {class_weights}')

class_weights_all = class_weights[target_list]

weighted_sampler = WeightedRandomSampler(weights=class_weights_all, num_samples=len(class_weights_all), replacement=True)

# Adjustable paramters for our model

epoch = 600
batch_size = 128
learning_rate = 0.00025
num_features = len(x.columns)
num_classes = 8

# Create the loaders for our train, val, and test dataset

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=weighted_sampler)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)

# Create our model

class MulticlassClassification(nn.Module):
    def __init__(self, numFeature, numClass):
        super(MulticlassClassification, self).__init__()
        
        self.layer1 = nn.Linear(numFeature, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layerOut = nn.Linear(32, numClass) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchNorm1 = nn.BatchNorm1d(256)
        self.batchNorm2 = nn.BatchNorm1d(128)
        self.batchNorm3 = nn.BatchNorm1d(64)
        self.batchNorm4 = nn.BatchNorm1d(32)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        
        x = self.layer2(x)
        x = self.batchNorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.batchNorm3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.batchNorm4(x)
        x = self.relu(x)
        
        x = self.layerOut(x)
        
        return x

# Print our device and model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}\n')

model = MulticlassClassification(numFeature=num_features, numClass=num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f'Model:\n{model}')

# Create methods to evaluate our model such as accuracy, precision, and recall

def modelAcc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

def accWithin(y_pred, y_test, within = 1):
    correct_pred = np.sum(np.abs(y_pred - y_test) <= within)
    acc = correct_pred / len(y_test)
    
    return round(acc * 100, 2)

def precisionWithin(y_pred, y_test, label, within=1):
    y_pred_class = y_pred[y_pred == label]

    correct_pred = 0
    for i in range(len(y_test)):
        if (y_pred[i] == label and abs(y_pred[i] - y_test[i]) <= within):
            correct_pred += 1

    precision = correct_pred / len(y_pred_class)

    return round(precision * 100, 2)
    
def recallWithin(y_pred, y_test, label, within=1):
    y_test_class = y_test[y_test == label]

    correct_pred = 0
    for i in range(len(y_test)):
        if (y_test[i] == label and abs(y_pred[i] - y_test[i]) <= within):
            correct_pred += 1

    recall = correct_pred / len(y_test_class)

    return round(recall * 100, 2)

# Set up the data structures we will use during training to store our results

accuracy_data = {
    'train': [],
    "val": []
}

loss_data = {
    'train': [],
    "val": []
}

# Train our model

print("Begin training.")

for e in tqdm(range(0, epoch)):
    
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0

    model.train()
    for x_train_batch, y_train_batch in train_loader:
        x_train_batch, y_train_batch = x_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_train_pred = model(x_train_batch)
        
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = modelAcc(y_train_pred, y_train_batch)
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
        
        
    # VALIDATION    
    with torch.no_grad():
        
        val_epoch_loss = 0
        val_epoch_acc = 0
        
        model.eval()
        for x_val_batch, y_val_batch in val_loader:
            x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = model(x_val_batch)
                        
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = modelAcc(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()

    accuracy_data['train'].append(train_epoch_acc/len(train_loader))
    accuracy_data['val'].append(val_epoch_acc/len(val_loader))
    loss_data['train'].append(train_epoch_loss/len(train_loader))
    loss_data['val'].append(val_epoch_loss/len(val_loader))
                              
    if (e % 25 == 0):
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')

# Graph our accuracy

train_val_acc_df = pd.DataFrame.from_dict(accuracy_data).reset_index().melt(id_vars=['index']).rename(columns={'index':'epochs'})
train_val_loss_df = pd.DataFrame.from_dict(loss_data).reset_index().melt(id_vars=['index']).rename(columns={'index':'epochs'})

x = []
for i in range(0, epoch // 25):
    x.append(i * 25)

# Plot the dataframes
plt.figure(figsize=(4, 3))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Epoch vs Train & Val Accuracy')
plt.plot(x, list(train_val_acc_df[(train_val_acc_df['variable'] == 'train') & (train_val_acc_df.index % 25 == 0)]['value']), label='Train')
plt.plot(x, list(train_val_acc_df[(train_val_acc_df['variable'] == 'val') & (train_val_acc_df.index % 25 == 0)]['value']), label='Val')
plt.legend()

plt.show()

# Graph our loss

plt.figure(figsize=(4, 3))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Train & Val Loss')
plt.plot(x, list(train_val_loss_df[(train_val_loss_df['variable'] == 'train') & (train_val_loss_df.index % 25 == 0)]['value']), label='Train')
plt.plot(x, list(train_val_loss_df[(train_val_loss_df['variable'] == 'val') & (train_val_loss_df.index % 25 == 0)]['value']), label='Val')
plt.legend()
plt.show()

# Use our model on our testing set

y_pred_list = []
with torch.no_grad():
    model.eval()
    for x_batch, _ in test_loader:
        x_batch = x_batch.to(device)
        y_test_pred = model(x_batch)
        _, y_pred_tags = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

# Create a confusion matrix based on the testing set

confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list)).rename(columns=labelsToCoverage, index=labelsToCoverage)
ax = plt.subplot()
sns.heatmap(confusion_matrix_df, annot=True, ax=ax)
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix');

print(classification_report(y_test, y_pred_list))

print(f'Accuracy: {accWithin(np.array(y_pred_list), y_test)}')
# Print a header row
print('{:<10} {:<15} {:<10}'.format('Label', 'Precision', 'Recall'))
for i in range(0, 8):    
    print('{:<10} {:<15} {:<10}'.format(i, precisionWithin(np.array(y_pred_list), y_test, i), recallWithin(np.array(y_pred_list), y_test, i)))

torch.save(model.state_dict(), 'my_model.pth')

future_df = data_df[data_df['Year'] == 2021]
future_df.head()

future_df['Prev Coverage'] = future_df['Coverage']
future_df.head()

x_future = future_df.iloc[:, list(range(3, 3)) + list(range(5, future_df.shape[1]))]
x_future.head()

x_future = scale.transform(x_future)
x_future, y_future = np.array(x_future), np.zeros(len(x_future))

print(len(x_future))
print(len(y_future))

future_dataset = createDataset(torch.from_numpy(x_future).float(), torch.from_numpy(y_future).long())
future_loader = DataLoader(dataset=future_dataset, batch_size=1)

# Test our model on our future dataset

y_future_pred_list = []
with torch.no_grad():
    model.eval()
    for x_batch, _ in future_loader:
        x_batch = x_batch.to(device)
        y_pred = model(x_batch)
        _, y_pred_tags = torch.max(y_pred, dim = 1)
        y_future_pred_list.append(y_pred_tags.cpu().numpy())
y_future_pred_list = [a.squeeze().tolist() for a in y_future_pred_list]

print(y_future_pred_list)

future_df['Coverage'] = y_future_pred_list
future_df.head()

danger_data = {'Avg Temp': [], 'Med Temp': [], 'Max Temp': [], 'Min Temp': []}
safe_data = {'Avg Temp': [], 'Med Temp': [], 'Max Temp': [], 'Min Temp': []}
for index, row in future_df.iterrows():
    if (row['Prev Coverage'] - row['Coverage'] > 0 or row['Coverage'] <= 3):
        # print(row)
        danger_data['Avg Temp'].append(row['Avg Temp'])
        danger_data['Med Temp'].append(row['Med Temp'])
        danger_data['Max Temp'].append(row['Max Temp'])
        danger_data['Min Temp'].append(row['Min Temp'])
    else:
        safe_data['Avg Temp'].append(row['Avg Temp'])
        safe_data['Med Temp'].append(row['Med Temp'])
        safe_data['Max Temp'].append(row['Max Temp'])
        safe_data['Min Temp'].append(row['Min Temp'])

print(f"Average Safe Temp: {np.sum(safe_data['Avg Temp']) / len(safe_data['Avg Temp'])}")
print(f"Median Safe Temp: {np.sum(safe_data['Med Temp']) / len(safe_data['Med Temp'])}")
print(f"Max Safe Temp: {np.sum(safe_data['Max Temp']) / len(safe_data['Max Temp'])}")
print(f"Min Safe Temp: {np.sum(safe_data['Min Temp']) / len(safe_data['Min Temp'])}")
print()
print(f"Average Danger Temp: {np.sum(danger_data['Avg Temp']) / len(danger_data['Avg Temp'])}")
print(f"Median Danger Temp: {np.sum(danger_data['Med Temp']) / len(danger_data['Med Temp'])}")
print(f"Max Danger Temp: {np.sum(danger_data['Max Temp']) / len(danger_data['Max Temp'])}")
print(f"Min Danger Temp: {np.sum(danger_data['Min Temp']) / len(danger_data['Min Temp'])}")

plt.figure(figsize=(4, 4))
plt.scatter(safe_data['Max Temp'], safe_data['Med Temp'], label='Safe')
plt.scatter(danger_data['Max Temp'], danger_data['Med Temp'], label='Danger')
plt.xlabel('Avg. Temp (℃)')
plt.ylabel('Max Temp (℃)')
plt.title('Safe Zones vs. Danger Zones')
plt.legend()
plt.show()
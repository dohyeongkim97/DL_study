#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


print(torch.tensor([1, 2, 3]))


# In[3]:


tensor = torch.rand(1, 2)


# In[4]:


tensor


# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
import scipy 
from scipy import stats
import matplotlib.pyplot as plt


# In[6]:


men = stats.norm.rvs(loc = 175, scale=10, size=500)
women = stats.norm.rvs(loc = 160, scale=10, size=500)


# In[7]:


x = np.concatenate([men, women])


# In[8]:


y = ['man'] * len(men) + ['women']*len(women)


# In[9]:


y


# In[10]:


df = pd.DataFrame(list(zip(x, y)), columns=['x', 'y'])


# In[11]:


sns.kdeplot(data=df, x='x', hue='y')


# In[12]:


stats, p_v = stats.ttest_ind(men, women, equal_var=True)
print(stats)
print(p_v)


# In[13]:


x = np.array([[i] for i in range(1, 31)])


# In[14]:


y = np.array([[0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1], [8.46], [9.5], [10.67], [11.16], [14], [11.83],
             [14.4], [14.25], [16.2], [16.32], [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]])


# In[15]:


weight = 0
bias = 0
learning_rate=0.005


# In[16]:


for epoch in range(10000):
    y_hat = weight*x+bias
    cost = ((y-y_hat)**2).mean()
    
    weight = weight - learning_rate*((y_hat-y)*x).mean()
    bias = bias - learning_rate*(y_hat-y).mean()
    
    if (epoch + 1)%100 == 0:
        print(f"Epoch : {epoch+1:4d}, weight : {weight:.3f}, bias : {bias:.3f}, cost : {cost:.3f}")


# In[17]:


import torch
from torch import optim


# In[18]:


x = torch.FloatTensor(x)
y = torch.FloatTensor(y)


# In[19]:


weight = torch.zeros(1, requires_grad = True)
bias = torch.zeros(1, requires_grad=True)
learning_rate = 0.001


# In[20]:


optimizer = optim.SGD([weight, bias], lr=learning_rate)


# In[21]:


for epoch in range(10000):
    hypothesis = x*weight+bias
    cost = torch.mean((hypothesis - y)**2)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if (epoch + 1)%500 == 0:
        print(f"Epoch : {epoch+1:4d}, weight : {weight.item():.3f}, bias : {bias.item():.3f}, cost : {cost:.3f}")


# In[22]:


from torch import nn


# In[23]:


model = nn.Linear(1, 1, bias=True)
criterion = nn.MSELoss()
learning_rate = 0.001


# In[24]:


help(nn.MSELoss)


# In[25]:


model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10000):
    output = model(x)
    cost = criterion(output, y)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if (epoch + 1)%500 == 0:
        print(f"Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, cost : {cost:.3f}")


# In[26]:


from torch.utils.data import TensorDataset, DataLoader


# In[27]:


train_x = torch.FloatTensor([
    [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]
])

train_y = torch.FloatTensor([
    [0.1, 1.5], [1, 2.8], [1.9, 4.1], [2.8, 5.4], [3.7, 6.7], [4.6, 8]
])


# In[28]:


train_dataset = TensorDataset(train_x, train_y)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last = True)


# In[29]:


model = nn.Linear(2, 2, bias=True)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


# In[30]:


for epoch in range(20000):
    cost = 0.0
    
    for batch in train_dataloader:
        x, y = batch
        output = model(x)
        
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        cost += loss
        
    cost = cost/len(train_dataloader)
    
    if (epoch + 1)%500 == 0:
        print(f"Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, cost : {cost:.3f}")


# In[34]:


from torch.nn import functional as F


# In[35]:


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


# In[36]:


import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader


# In[ ]:





# In[39]:


class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.x = df.iloc[:, 0].values
        self.y = df.iloc[:, 0].values
        self.length = len(df)
        
    def __getitem__(self, index):
        x = torch.FloatTensor([self.x[index]  ** 2, self.x[index]])
        y = torch.FloatTensor([self.y[index]])
        return x, y
    
    def __len__(self):
        return self.length


# In[38]:


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)
        
    def forward(self, x):
        x = self.layer(x)
        return x


# In[40]:


from torch.utils.data import random_split


# In[41]:


dataset = CustomDataset('./non_linear.csv')
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
validation_size = int(dataset_size * 0.1)
test_size = dataset_size - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last = True)
validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)


# In[43]:


with torch.no_grad():
    model.eval()
    
    for x, y in validation_dataloader :
        
        outputs = model(x)
        print(f'outputs : {outputs}')


# 무작위 분리 함수(random_split)
# ```python
# subset = torch.utils.data.random_split(
#     dataset,
#     lengths(list),
#     generator
# )
# ```

# # classification

# In[45]:


class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        
        self.x1 = df.iloc[:, 0].values
        self.x2 = df.iloc[:, 1].values
        self.x3 = df.iloc[:, 2].values
        self.y = df.iloc[:, 3].values
        self.length = len(df)
        
    def __getitem__(self, index):
        x = torch.FloatTensor([self.x1[index], self.x2[index], self.x3[index]])
        y = torch.FloatTensor([self.y[index]])
        return x, y
    
    def __len__(self):
        return self.length
    
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
    
    def forward(self, y):
        x = self.layer(x)
        return x


# In[47]:


criterion = nn.BCELoss()


# In[ ]:





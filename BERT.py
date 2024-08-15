#!/usr/bin/env python
# coding: utf-8

# ## BERT
# 
# (Bidirectional Encoder Representations from Transformers)
# 
# 입력 시퀀스를 양측에서 처리하여 이전과 이후의 단어를 모두 참조하며 단어의 의미를 파악.
# 
# 기존 모델들보다 정확하게 문맥을 파악하고 다양한 자연어 처리 작업에서 높은 성능을 보임.
# 
# 대규모 데이터를 통해 사전 학습되어 있으므로 전이 학습에 주로 활용. 
# 
# BERT는 일부나 전체를 다른 작업에서 재사용하여 적은 데이터로도 높은 정확도를 달성. 

# ### Masked Langauge Modeling, MLM
# 
# 입력 문장에서 임의로 일부 단어를 마스킹, 해당 단어를 예측.

# ### Next Sentence Prediction, NSP
# 
# 두 문장이 주어졌을 때, 두 번째 문장이 첫 번째 문장의 다음에 오는 문장인지의 여부 판단.

# BERT 모델의 토큰
# 
# CLS / SEP / MASK
# 

# CLS:
# 
# 입력 문장의 시작. 해당 토큰을 사용해 문장 분류 작업을 위한 정보를 제공.
# 
# 모델은 문장이 어떤 유형의 것인지 미리 파악 가능하게 됨. 
# 

# SEP:
# 
# 입력 문장 내에서 문장의 구분을 위해 사용. 문장 분률 작업에서 복수 문장을 입력으로 받을 때, SEP 토큰을 통해 문장을 구분.

# MASK:
# 
# 입력 문장 내에서 임의로 선택된 단어를 가리키는 토큰. 주어진 문장에서 단어를 가린 후 모델의 학습과 예측에 활용. 

# In[1]:


import numpy as np
import pandas as pd
from Korpora import Korpora


# In[2]:


corpus = Korpora.load('nsmc')
df = pd.DataFrame(corpus.test).sample(20000, random_state=42)


# In[3]:


train, valid, test = np.split(
    df.sample(frac = 1, random_state = 42), [int(0.6*len(df)), int(0.8*len(df))]
)


# In[4]:


print(train.head(5).to_markdown())


# In[5]:


import sys
sys.path
sys.path.append('C:/Users/dohyeong/miniconda3/Lib/site-packages/')

import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch import optim
from transformers import BertForSequenceClassification
from torch import nn
import math
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


# In[8]:


def make_dataset(data, tokenizer, device):
    tokenized = tokenizer(
        text = data.text.tolist(),
        padding= 'longest',
        truncation = True,
        return_tensors = 'pt'
    )
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    labels = torch.tensor(data.label.values, dtype=torch.long).to(device)
    return TensorDataset(input_ids, attention_mask, labels)


# In[9]:


def get_dataloader(dataset, sampler, batch_size):
    data_sampler = sampler(dataset)
    dataloader = DataLoader(dataset, sampler = data_sampler, batch_size = batch_size)
    return dataloader


# In[10]:


epochs = 5
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-multilingual-cased',
    do_lower_case = False
)


# In[11]:


device


# In[12]:


train_dataset = make_dataset(train, tokenizer, device)
train_dataloader = get_dataloader(train_dataset, RandomSampler, batch_size)

valid_dataset = make_dataset(valid, tokenizer, device)
valid_dataloader = get_dataloader(valid_dataset, RandomSampler, batch_size)

test_dataset = make_dataset(test, tokenizer, device)
test_dataloader = get_dataloader(test_dataset, RandomSampler, batch_size)


# In[13]:


print(train_dataset[0])


# In[15]:


model = BertForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path='bert-base-multilingual-cased',
    num_labels = 2
).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)


# In[26]:


def calc_accuracy(preds, labels):
    pred_flat = np.argmax(preds.cpu().numpy(), axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[27]:


def train(model, optimizer, dataloader):
    model.train()
    train_loss = 0.0
    
    for input_ids, attention_mask, labels in dataloader:
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )
        
        loss = outputs.loss
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss = train_loss / len(dataloader)
    return train_loss


# In[28]:


def evaluation(model, dataloader):
    with torch.no_grad():
        model.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss, val_accuracy = 0.0, 0.0
        
        for input_ids, attention_mask, labels in dataloader:
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels
            )
            logits = outputs.logits
            
            loss = criterion(logits, labels)
            logtis = logits.detach().cpu().numpy()
            labels_ids = labels.to('cpu').numpy()
            accuracy = calc_accuracy(logits, labels_ids)
            
            val_loss += loss
            val_accuracy += accuracy
            
        val_loss = val_loss/len(dataloader)
        val_accuracy = val_accuracy / len(dataloader)
        return val_loss, val_accuracy


# In[29]:


best_loss = 10000
for epoch in range(epochs):
    train_loss = train(model, optimizer, train_dataloader)
    val_loss, val_accuracy = evaluation(model, valid_dataloader)
    print(f'Epoch: {epoch+1} train loss: {train_loss:.4f} val loss: {val_loss:.4f} val_acc: {val_accuracy:.4f}')
    
    if val_loss < best_loss:
        best_loss = val_loss


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





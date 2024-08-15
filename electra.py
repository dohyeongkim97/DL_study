#!/usr/bin/env python
# coding: utf-8

# ## ELECTRA(Efficiently Learning an Encoder that Classifies Token Replacements Accurately)
# 
# BART / BERT 등에서 사용되는 MLM(Masking Language Modeling) 기법의 입력 마스킹 대신 Generator(생성자)와 Discriminator(판별자)를 사용하는 방식.

# 생성자와 판별자를 학습하므로 생성적 적대 신경망(GAN)과 유사한 방식으로 학습이 수행됨.
# 
# 생성 모델은 실제 데이터와 비슷하게 토큰을 생성하여 다른 토큰으로 대체하고, 판별 모델이 생성 모델이 만든 데이터를 입력받아 어떤 데이터가 실제인지 생성 데이터인지를 구분.

# GAN 모델을 사용하여 이전 모델과 비교하여 더 효율적인 학습이 가능. 대규모 데이터세트에서 모델을 더 빠르게 학습 가능함. 생성 모델을 통해 토큰을 생성하므로 다양한 자연어 생성 작업에서 보다 자연스러운 문장을 생성.

# BERT 등에 비해 모델의 매개변수 수가 더 적어 더 빠른 실행과 더 적은 메모리 수요를 충족

# ### Pretrain

# ELECTRA의 generator와 discriminator는 tf encoder 구조를 따름. 생성자 모델은 입력 문장의 일부 토큰을 마스크, 마스크 처리된 토큰이 어떤 토큰이었는지 예측.
# 
# 판별자 모델은 입력 토큰이 원본 문장 토큰인지를 예측하며 학습.

# 이러한 학습 방식을 RTD(Replaced Token Detection)이라 칭함.

# 사전 학습이 완료되면 생성 모델을 사용하지 아니하고 다운스트림 작업을 수행.

# In[1]:


import sys
sys.path.append("C:/Users/dohyeong/miniconda3/Lib/site-packages/")

sys.path

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

import numpy as np
import pandas as pd
from Korpora import Korpora
import torch
from transformers import ElectraTokenizer


corpus = Korpora.load('nsmc')
df = pd.DataFrame(corpus.test).sample(20000, random_state=42)
train, valid, test = np.split(
    df.sample(frac=1, random_state=42), [int(0.6*len(df)), int(0.8*len(df))]
)


# In[2]:


def make_dataset(data, tokenizer, device):
    tokenized = tokenizer(
        text = data.text.tolist(),
        padding = 'longest',
        truncation = True,
        return_tensors = 'pt'
    )
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    labels = torch.tensor(data.label.values, dtype=torch.long).to(device)
    return TensorDataset(input_ids, attention_mask, labels)


# In[3]:


def get_dataloader(dataset, sampler, batch_size):
    data_sampler = sampler(dataset)
    dataloader = DataLoader(dataset, sampler = data_sampler, batch_size = batch_size)
    return dataloader


# In[4]:


epochs = 5
batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[5]:


device


# In[6]:


tokenizer = ElectraTokenizer.from_pretrained(
    pretrained_model_name_or_path= 'monologg/koelectra-base-v3-discriminator',
    do_lower_case = False
)


# In[7]:


train_dataset = make_dataset(train, tokenizer, device)
train_dataloader = get_dataloader(train_dataset, RandomSampler, batch_size)

valid_dataset = make_dataset(valid, tokenizer, device)
valid_dataloader = get_dataloader(valid_dataset, RandomSampler, batch_size)

test_dataset = make_dataset(test, tokenizer, device)
test_dataloader = get_dataloader(test_dataset, RandomSampler, batch_size)


# In[8]:


print(train.head(5).to_markdown())


# In[9]:


print(train_dataset[0])


# HuggingFace에서 electra용 모델 제공. 영어 텍스트용 ELECTRA / 한국어 텍스트용 KoELECTRA
# 
# 영문은 google/electra-small | google/electra-base | google/electra-large 로 적용, 국문은 monologg/koelectra-small-v3 | monologg/koelectra-base-v3 으로 적용

# ELECTRA는 판별 모델만을 통해 다운스트림 작업을 수행하므로, koelectra-base 모델의 판별 모델인 monologg/koelectra-base-discriminator 모델을 로드.
# 
# 생성 모델을 불러와야 하는 경우라면, monologg/koelectra-base-generator를 통해 로드.

# In[10]:


from torch import optim
from transformers import ElectraForSequenceClassification


# In[11]:


model = ElectraForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path = 'monologg/koelectra-base-v3-discriminator',
    num_labels = 2
).to(device)


# In[12]:


optimizer = optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)


# In[13]:


import numpy as np
from torch import nn


# In[14]:


def calc_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[15]:


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


# In[16]:


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
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            accuracy = calc_accuracy(logits, label_ids)
            
            val_loss += loss
            val_accuracy += accuracy
            
        val_loss = val_loss / len(dataloader)
        val_accuracy = val_accuracy / len(dataloader)
        return val_loss, val_accuracy


# In[ ]:


best_loss = 10000
for epoch in range(epochs):
    train_loss = train(model, optimizer, train_dataloader)
    val_loss, val_accuracy = evaluation(model, valid_dataloader)
    print(f'epoch: {epoch+1}, train: {train_loss:.4f}, val: {val_loss:.4f}, val_ac: {val_accuracy:.4f}')
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'electra_model.pt')
        print()
        print('model saved')
        print()


# ### GLUE(General Language Understanding Evaluation) Benchmark dataset

# 머신러닝 알고리즘 성능 평가를 위한 표준 데이터셋. 고품질 데이터와 레이블된 결과를 포함. 알고리즘 성능 비교를 위해 공개적인 사용이 가능.
# 
# 문장 수준 / 문서 수준의 이해력을 평가하는 데이터세트. 문장 분류, 유사도 계산, 자연어 추론, 질의응답 등 총 11가지 과제.

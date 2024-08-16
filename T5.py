#!/usr/bin/env python
# coding: utf-8

# ## T5(Text-toText Transfer Transformer)

# 기존 자연어 처리 모델은 대부분 입력 문장을 벡터나 행렬로 변환한 뒤, 이를 통해 출력 문장ㅇ르 생성하는 방식이거나, 출력값이 클래스나 입력값의 일부를 반환하는 형식으로 동작.

# T5는 출력을 모두 토큰 시퀀스로 처리하는 Text to Text structure.
# 
# 입력과 출력의 형태를 자유로이 다룰 수 있으며, 구조상 유연성과 확장성이 뛰어남.

# 문장마다 마스크 토큰을 사용하는 Sentinel Token을 사용. <extra_id_0> 이나 <extra_id_1> 처럼, 0부터 99개의 기본값.

# In[11]:


import numpy as np
from datasets import load_dataset


# In[12]:


news = load_dataset('argilla/news-summary', split='test')
df = news.to_pandas().sample(5000, random_state=42)[['text', 'prediction']]
df['text'] = 'summarize: ' + df['text']
df['prediction'] = df['prediction'].map(lambda x: x[0]['text'])
train, valid, test = np.split(
    df.sample(frac = 1, random_state = 42), [int(0.6*len(df)), int(0.8*len(df))]
)


# In[13]:


train


# In[14]:


train['text'][9209]


# In[15]:


train['prediction'][9209]


# In[16]:


import sys
sys.path.append("C:/Users/dohyeong/miniconda3/Lib/site-packages/")


# In[17]:


import torch
from transformers import T5Tokenizer
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler


# In[18]:


from torch import optim


# In[19]:


def make_dataset(data, tokenizer, device):
    source = tokenizer(
        text = data.text.tolist(),
        padding='max_length',
        max_length=128,
        pad_to_max_length=True,
        truncation=True,
        return_tensors='pt'
    )
    
    target = tokenizer(
        text = data.prediction.tolist(),
        padding='max_length',
        max_length=128,
        pad_to_max_length= True,
        truncation = True,
        return_tensors = 'pt'
    )
    
    source_ids = source['input_ids'].squeeze().to(device)
    source_mask = source['attention_mask'].squeeze().to(device)
    target_ids = target['input_ids'].squeeze().to(device)
    target_mask = target['attention_mask'].squeeze().to(device)
    return TensorDataset(source_ids, source_mask, target_ids, target_mask)


# In[20]:


def get_dataloader(dataset, sampler, batch_size):
    data_sampler = sampler(dataset)
    dataloader = DataLoader(dataset, sampler = data_sampler, batch_size = batch_size)
    return dataloader


# In[21]:


epochs = 3
batch_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[22]:


device


# In[23]:


tokenizer = T5Tokenizer.from_pretrained(
    pretrained_model_name_or_path= 't5-small'
)


# In[24]:


train_dataset = make_dataset(train, tokenizer, device)
train_dataloader = get_dataloader(train_dataset, RandomSampler, batch_size)


# In[25]:


valid_dataset = make_dataset(valid, tokenizer, device)
valid_dataloader = get_dataloader(valid_dataset, RandomSampler, batch_size)


# In[26]:


test_dataset = make_dataset(test, tokenizer, device)
test_dataloader = get_dataloader(test_dataset, RandomSampler, batch_size)


# In[27]:


print(next(iter(train_dataloader)))


# In[28]:


from torch import optim
from transformers import T5ForConditionalGeneration


# In[29]:


model = T5ForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path= 't5-small'
).to(device)


# In[30]:


optimizer = optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)


# In[31]:


import numpy as np
from torch import nn


# In[32]:


def train(model, optimizer, dataloader):
    model.train()
    train_loss = 0.0
    
    for source_ids, source_mask, target_ids, target_mask in dataloader:
        decoder_input_ids = target_ids[:, :-1].contiguous()
        labels = target_ids[:, 1:].clone().detach()
        labels[target_ids[:, 1:] == tokenizer.pad_token_id] = -100
        
        outputs = model(
            input_ids = source_ids,
            attention_mask = source_mask,
            decoder_input_ids = decoder_input_ids,
            labels = labels
        )
        
        loss = outputs.loss
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss = train_loss / len(dataloader)
    return train_loss


# In[33]:


def evaluation(model, dataloader):
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        
        for source_ids, source_mask, target_ids, target_mask in dataloader:
            decoder_input_ids = target_ids[:, :-1].contiguous()
            labels = target_ids[:, 1:].clone().detach()
            labels[target_ids[:, 1:] == tokenizer.pad_token_id] = -100
            
            outputs = model( 
                input_ids = source_ids,
                attention_mask = source_mask,
                decoder_input_ids = decoder_input_ids,
                labels = labels,
            )
            
            loss = outputs.loss
            val_loss += loss
            
        val_loss = val_loss / len(dataloader)
        return val_loss


# In[34]:


best_loss = 10000

for epoch in range(epochs):
    train_loss = train(model, optimizer, train_dataloader)
    val_loss = evaluation(model, valid_dataloader)
    print(f"epoch: {epoch+1}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), './t5generator.pt')
        print()
        print('model saved')


# In[35]:


model.eval()


# In[37]:


with torch.no_grad():
    for source_ids, source_mask, target_ids, target_mask in test_dataloader:
        generated_ids = model.generate(
            input_ids = source_ids, 
            attention_mask = source_mask,
            max_length = 128,
            num_beams = 3,
            repetition_penalty = 2.5,
            length_penalty = 1.0,
            early_stopping = True
        )
        
        for generated, target in zip(generated_ids, target_ids):
            pred = tokenizer.decode(
                generated, skip_special_tokens= True, clean_up_tokenization_spaces= True
            )
            actual = tokenizer.decode(
                target, skip_special_tokens=True, clean_up_tokenization_spaces= True,
            )
            
            print('generated_headline_text: ', pred)
            print('actual_headline: ', actual)
            print('')
        break


# In[ ]:





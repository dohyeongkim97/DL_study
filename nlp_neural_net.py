#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import Korpora
from Korpora import Korpora
import numpy as np
import pandas as pd
import gensim


# In[2]:


from gensim.models import FastText


# In[3]:


corpus = Korpora.load('kornli')
corpus_texts = corpus.get_all_texts() + corpus.get_all_pairs()


# In[5]:


tokens = [sentence.split() for sentence in corpus_texts]


# In[8]:


fasttext = FastText(
    sentences = tokens,
    vector_size = 128,
    window = 5,
    min_count = 5,
    sg = 1,
    epochs = 3,
    min_n = 2,
    max_n = 6
)


# In[10]:


oov_token = '사랑해요'
oov_vector = fasttext.wv[oov_token]


# In[11]:


print(oov_token in fasttext.wv.index_to_key)
print(fasttext.wv.most_similar(oov_vector, topn=5))


# # RNN

# ```
# rnn = torch.nn.RNN(
#     input_size,
#     hidden_size,
#     num_layers = 1,
#     nomlinearity='tanh',
#     bias=False,
#     batch_first = True,
#     dropout = 0,
#     bidirectional = False
# )
# ```

# In[13]:


import torch
from torch import nn


# In[21]:


input_size = 128
output_size = 256
num_layers = 3
bidirectional = True


# In[22]:


model = nn.RNN(
    input_size = input_size,
    hidden_size = output_size,
    num_layers = num_layers,
    nonlinearity='tanh',
    batch_first = True,
    bidirectional=bidirectional,
)


# In[23]:


batch_size=4
sequence_len=6


# In[24]:


inputs = torch.randn(batch_size, sequence_len, input_size)


# In[25]:


h0 = torch.rand(num_layers * (int(bidirectional)+1), batch_size, output_size)


# In[26]:


outputs, hidden = model(inputs, h0)


# In[27]:


print(outputs.shape)
print(hidden.shape)


# # LSTM

# Long Short Term Memory: RNN 모델이 갖던 기억력 부족과 Gradient Vanishing 문제를 해결

# RNN 모델은 장기 의존성 문제(Long Term Dependencies) 문제가 발생 가능. 활성화함수로 사용되는 tanh 함수나 ReLU 함수 특성으로 인해 역전파 과정에서 기울기 소실이나 폭주도 발생 가능함.

# LSTM 모델은 순환 싱경망과 비슷한 구조를 가지나, Memory cell과 Gate 구조의 도입으로 상기한 문제를 해결

# ```
# lstm = torch.nn.LSTM(
#     input_size,
#     hidden_size,
#     num_layers=1,
#     bias=True,
#     batch_first=True,
#     dropout=0,
#     bidirectional=False,
#     proj_size=0
# )
# ```

# In[29]:


import torch
from torch import nn


# In[48]:


input_size=128
output_size=256
num_layers = 3
bidirectional=True
proj_size=64

model = nn.LSTM(
    input_size=input_size,
    hidden_size=output_size,
    num_layers=num_layers,
    batch_first=True,
    bidirectional=bidirectional,
    proj_size=proj_size
)

batch_size=4
sequence_len=6

inputs=torch.randn(batch_size, sequence_len, input_size)
h0=torch.rand(
    num_layers * (int(bidirectional)+1),
    batch_size,
    proj_size if proj_size > 0 else output_size,
)
c0 = torch.rand(num_layers * (int(bidirectional)+1), batch_size, output_size)

outputs, (hn, cn) = model(inputs, (h0, c0))


# In[49]:


print(outputs.shape)


# In[51]:


print(hn.shape)
print(cn.shape)


# # P/N classification model by using RNN and LSTM

# In[52]:


class sentence_classifier(nn.Module):
    def __init__(
        self,
        n_vocab,
        hidden_dim,
        embeedding_dim,
        n_layers,
        dropout=0.5,
        bidirectional=True,
        model_type='lstm'
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim = embedding_dim,
            padding_idx = 0
        )
        if model_type == 'rnn':
            self.model = nn.RNN(
                input_size = embedding_dim,
                hidden_size = hidden_dim,
                num_layers = n_layers,
                bidirectional = bidirectional,
                dropout = dropout,
                batch_first = True
            )
        
        elif model_type == 'lstm':
            self.model = nn.LSTM(
                input_size = embedding_dim,
                hidden_size = hidden_dim,
                num_layers = n_layers,
                bidirectional = bidirectional,
                dropout = dropout,
                batch_first = True
            )
        
        if bidirectional:
            self.classifier = nn.Linear(hidden_dim*2, 1)
        else:
            self.classifier = nn.Linear(hidden_dim, 1)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        output, _ = self.model(embeddings)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        logits = self.classifier(last_output)
        return logits


# In[ ]:





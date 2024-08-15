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


# In[4]:


tokens = [sentence.split() for sentence in corpus_texts]


# In[5]:


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


# In[6]:


oov_token = '사랑해요'
oov_vector = fasttext.wv[oov_token]


# In[7]:


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

# In[8]:


import torch
from torch import nn


# In[9]:


input_size = 128
output_size = 256
num_layers = 3
bidirectional = True


# In[10]:


model = nn.RNN(
    input_size = input_size,
    hidden_size = output_size,
    num_layers = num_layers,
    nonlinearity='tanh',
    batch_first = True,
    bidirectional=bidirectional,
)


# In[11]:


batch_size=4
sequence_len=6


# In[12]:


inputs = torch.randn(batch_size, sequence_len, input_size)


# In[13]:


h0 = torch.rand(num_layers * (int(bidirectional)+1), batch_size, output_size)


# In[14]:


outputs, hidden = model(inputs, h0)


# In[15]:


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

# In[16]:


import torch
from torch import nn


# In[17]:


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


# In[18]:


print(outputs.shape)


# In[19]:


print(hn.shape)
print(cn.shape)


# # P/N classification model by using RNN and LSTM

# In[20]:


class sentence_classifier(nn.Module):
    def __init__(
        self,
        n_vocab,
        hidden_dim,
        embedding_dim,
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


# In[21]:


import pandas as pd
from Korpora import Korpora


# In[22]:


corpus = Korpora.load('nsmc')
corpus_df = pd.DataFrame(corpus.test)


# In[23]:


train = corpus_df.sample(frac=0.9, random_state=42)
test = corpus_df.drop(train.index)


# In[24]:


print(train.head(5).to_markdown())
print(f'train_size: {len(train)}')
print(f'test_size: {len(test)}')


# In[25]:


from konlpy.tag import Okt
from collections import Counter


# In[26]:


def build_vocab(corpus, n_vocab, special_tokens):
    counter = Counter()
    for tokens in corpus:
        counter.update(tokens)
        vocab = special_tokens
    for token, count in counter.most_common(n_vocab):
        vocab.append(token)

    return vocab

tokenizer = Okt()
print('okt done')
train_tokens = [tokenizer.morphs(review) for review in train.text]
test_tokens = [tokenizer.morphs(review) for review in test.text]

vocab = build_vocab(corpus=train_tokens, n_vocab=5000, special_tokens=['<pad>', '<unk>'])
token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for idx, token in enumerate(vocab)}


# In[27]:


print(vocab[:10])
print(len(vocab))


# # int encoding and padding

# In[28]:


import numpy as np


# In[29]:


def pad_sequences(sequences, max_length, pad_value):
    result = list()
    for sequence in sequences:
        sequence = sequence[:max_length]
        pad_length = max_length - len(sequence)
        padded_sequence = sequence + [pad_value] * pad_length
        result.append(padded_sequence)
    return np.asarray(result)


# In[30]:


unk_id = token_to_id['<unk>']
train_ids = [
    [token_to_id.get(token, unk_id) for token in review] for review in train_tokens
]
test_ids = [
    [token_to_id.get(token, unk_id) for token in review] for review in test_tokens
]

max_length = 32
pad_id = token_to_id['<pad>']
train_ids = pad_sequences(train_ids, max_length, pad_id)
test_ids = pad_sequences(test_ids, max_length, pad_id)


# In[31]:


print(train_ids[0])


# In[32]:


print(test_ids[0])


# In[33]:


from torch.utils.data import TensorDataset, DataLoader

train_ids = torch.tensor(train_ids)
test_ids = torch.tensor(test_ids)

train_labels = torch.tensor(train.label.values,  dtype = torch.float32)
test_labels = torch.tensor(test.label.values, dtype=torch.float32)

train_dataset = TensorDataset(train_ids, train_labels)
test_dataset = TensorDataset(test_ids, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


# In[34]:


from torch import optim


# In[35]:


n_vocab = len(token_to_id)


# In[64]:


hidden_dim = 64
embedding_dim = 128
n_layers = 2

classifier = sentence_classifier(
    n_vocab = n_vocab,
    hidden_dim = hidden_dim,
    embedding_dim = embedding_dim,
    n_layers = n_layers
)


# In[112]:


device = 'cuda'


# In[113]:


criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.RMSprop(classifier.parameters(), lr=0.001)


# In[142]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[143]:


device


# In[223]:


def train(model, datasets, criterion, device, optimizer, interval):
    model.train()
    losses = list()
    for step, (input_ids, labels) in enumerate(datasets):
        input_ids = input_ids.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        logits = model(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % interval == 0:
            print(f'train_loss {step}: {np.mean(losses)}')
            


# In[224]:


def test(model, datasets, criterion, device):
    model.eval()
    losses = []
    corrects = []
    
    with torch.no_grad():
        for step, (input_ids, labels) in enumerate(datasets):
            input_ids = input_ids.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            logits = model(input_ids)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            
            predictions = (logits > 0.5).float()
            correct = (predictions == labels).float().sum()
            corrects.append(correct)
    
    avg_loss = sum(losses) / len(losses)
    accuracy = sum(corrects) / len(datasets.dataset)
    print(f'Test Loss: {avg_loss}, Test Accuracy: {accuracy}')


# In[225]:


epochs = 5
interval = 500


# In[226]:


device = 'cuda'


# In[227]:


model.to(device)


# In[228]:


classifier.to(device)


# In[236]:


for epoch in range(epochs):
    train(classifier, train_loader, criterion, device, optimizer, interval)
    test(classifier, test_loader, criterion, device)


# In[222]:


token_to_embedding = dict()
embedding_matrix = classifier.embedding.weight.detach().cpu().numpy()


# In[200]:


for word, emb in zip(vocab, embedding_matrix):
    token_to_embedding[word] = emb


# In[201]:


token = vocab[1000]
print(token, token_to_embedding[token])


# In[202]:


len(token_to_embedding['보고싶다'])


# In[203]:


token_to_embedding['보고싶다'].sum()


# In[204]:


from gensim.models import Word2Vec


# In[205]:


word2vec = Word2Vec.load('./word2vec.model')
init_embeddings = np.zeros((n_vocab, embedding_dim))


# In[206]:


embedding_dim


# In[207]:


len(init_embeddings)


# In[208]:


id_to_token


# In[209]:


for index, token in id_to_token.items():
    if token not in ['<pad>', '<unk>']:
        init_embeddings[index] = word2vec.wv[token]


# In[210]:


embedding_layer = nn.Embedding.from_pretrained(
    torch.tensor(init_embeddings, dtype=torch.float32)
)


# In[230]:


class sentence_classifier(nn.Module):
    def __init__(
        self,
        n_vocab,
        hidden_dim,
        embedding_dim,
        n_layers,
        dropout=0.5,
        bidirectional=True,
        model_type='lstm',
        pretrained_embedding=None  # 새로운 매개변수 추가
    ):
        super().__init__()
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_embedding, dtype=torch.float32)
            )
        else:
            self.embedding = nn.Embedding(
                num_embeddings=n_vocab,
                embedding_dim=embedding_dim,
                padding_idx=0
            )
        
        if model_type == 'rnn':
            self.model = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True
            )
        
        elif model_type == 'lstm':
            self.model = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True
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


# In[231]:


classifier = sentence_classifier(
    n_vocab = n_vocab,
    hidden_dim = hidden_dim,
    embedding_dim = embedding_dim,
    n_layers = n_layers,
    pretrained_embedding = init_embeddings,
).to(device)


# In[232]:


epochs = 5
interval = 500


# In[237]:


for epoch in range(epochs):
    train(classifier, train_loader, criterion, device, optimizer, interval)
    test(classifier, test_loader, criterion, device)


# In[ ]:





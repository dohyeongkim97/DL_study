#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys


# In[2]:


sys.path


# In[3]:


sys.path.append('C:/Users/dohyeong/miniconda3/Lib/site-packages/')


# In[4]:


import torch


# In[5]:


torch.cuda.is_available()


# # Self-Attention

# sequential processing이나 recurrent connections에 의존하지 않고, 입력 토큰 간의 거리를 직접 처리하고 이해할 수 있게끔 하는 Self Attention을 기반. 

# 대용량 세트에서 효율적. 기계 번역 등의 작업에서 적합. 장기적 종속성을 포함하는 작업에 주로 사용.

# Auto - Encoding, Auto - Regressive

# models :
# 
# BERT : 인코더, 오토 인코딩, 양방향 학습
# 
# GPT : 디코더, 자기회귀, 단방향 학습
# 
# BART : 인코더+디코더, 오토인코딩+자기회귀, 복합 학습
# 
# ELECTRA : 디코더+판별기, 오토인코딩+대체토큰탐지, 양방향 학습
# 
# T5 : 인코더+디코더, 오토인코딩+자기회귀+처리작업학습, 양방향 학습

# ## Transformers

# Attention Mechanism. Sequence Embedding을 표현.

# 기존 순환 싱경망 기반 모델보다 학습 속도가 빠르고 병렬 처리가 가능. 따라서, 대규모 데이터세트에서 높은 성능을 보임. 
# 
# 임베딩 과정에서 문장의 전체 정보를 고려하기 때문에, 문장 길이가 길어져도 성능을 유지.

# ### Multi Head Attention

# input sequence에서 query, key, value 벡터를 정의하여 입력 시퀀스들의 관계를 셀프어텐션하는 벡터 표현 방식. 
# 
# 해당 과정에서 쿼리와 각 키의 유사도를 계산, 해당 유사도를 가중치로 사용하여 값 벡터를 환산

# 순방향 신경망은 해당 과정에서 산출된 임베딩 벡터를 더욱 고도화하기 위해 사용. 해당 신경망은 복수 선형 계층으로 구성.
# 
# 순방향 신경망의 구조와 동일하게 입력 벡터와 가중치를 곱하고 편향을 더해 활성화 함수로 적용.

# 입력 시퀀스를 Source와 Target으로 나누어 처리. 영어를 한글로 번역하는 경우, 생성 언어인 한글을 target, 참조 언어인 영어를 source(본디 번역을 위해 만들어진 알고리즘이므로 위 구조가 타당)
# 
# 인코더는 source sequence data를 Positional Encoding된 입력 임베딩으로 표현하여 트랜스포머 블록의 출력 벡터를 생성. 해당 벡터는 입력 시퀀스 데이터의 관계를 잘 표현 가능하게 구성.

# 디코더도 인코더와 유사하게 transformer block으로 구성되어 있으나, Mastked Multi-Head Attention을 사용하여 타깃 시퀀스 데이터를 순차 생성. 

# ## 입력 임베딩 / 위치 인코딩

# transformer 모델에서 입력 시퀀스의 각 단어는 임베딩 처리되어 벡터 형태로 변환. Transformer 모델은 순환 신경망과 달리, 입 력 시퀀스를 병렬 구조로 처리하기 때문에, 단어의 순서 정보를 제공하지 아니함.
# 
# 따라서, 위치 정보를 임베딩 벡터에 추가하여 단어의 순서 정보를 모델에 반영. 위치 인코딩은 이를 위한 방식

# 위치 인코딩 벡터는 sin / cos 함수를 이용하여 생성. 이를 통해 임베딩 벡터와 위치 정보가 결합된 최종 입력 벡터를 생성.
# 
# 위치 인코딩 벡터를 추가하여 모델은 단어의 순서 정보를 학습 가능하게 됨.

# 위치 인코딩은 각 토큰의 위치를 각도로 표현하여 sin 함수와 cos 함수로 위치 인코딩 벡터를 계산. 

# In[6]:


import torch
import math
from torch import nn
from matplotlib import pyplot as plt


# In[7]:


class positional_encoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position*div_term)
        pe[:, 0, 1::2] = torch.cos(position*div_term)
        self.register_buffer('pe', pe) # 모델이 매개변수를 갱신하지 않도록 설정
        
    def forward(self, x):
        x = x+self.pe[:, x.size(0)]
        return self.dropout(x)


# In[8]:


encoding = positional_encoding(d_model = 128, max_len = 50)


# In[9]:


plt.pcolormesh(encoding.pe.numpy().squeeze(), cmap='RdBu')
plt.xlabel('embedding dimension')
plt.xlim((0, 128))
plt.ylabel('position')
plt.colorbar()
plt.show()


# ## 특수 토큰

# 단어 토큰이나 특수 토큰을 활용하여 문장을 표현. 해당 특수 토큰은 입력 시퀀스의 처음과 끝을 나타내거나, masking 영역으로 활용
# 
# e.g. 번역 모델에서는 디코더의 입력 시퀀스에서 현재 위치 이후의 토큰을 마스크해서 이전 토큰만을 참조.

# BOS: Beginning of Sentence
# 
# EOS: End of Sentence
# 
# UNK: Unknown
# 
# PAD: padding

# ## TF Encoder

# 입력 시퀀스를 받아 복수 계층으로 구성된 인코딩 계층을 거쳐 연산을 수행. 각 계층은 멀티 헤드 어텐션과 순방향 신경망으로 구성.
# 
# 인코더 계층에서 위치 정보 반영을 위해 위치 임베딩 벡터를 입력 벡터에 합산. 산출된 계층의 출력은 디코더 계층으로 전달. 

# TF 인코더는 위치 인코딩이 적용된 소스 데이터의 입력 임베딩을 입력받음. MHA 단계에서 입력 텐서 차원이 [N, s, d]라고 한다면, 입력 임베딩은 선형 변환을 통하여 세개의 임베딩 벡터를 생성. 각 Query(Q), Key(K), Value(V) 벡터로 정의

# Q: 현재 시점에서 참조하고자 하는 정보의 위치. 인코더의 각 시점마다 생성.
# 
# K: 쿼리 벡터와의 비교 대상. 입력 시퀀스에서 탐색되는 벡터. 인코더의 각 시점에서 생성
# 
# V: Q와 K로 생성된 어텐션 스코어를 얼마나 반영할지 설정하는 가중치.

# $\text{attention score}(v^q, v^k) = \text{softmax}\left(\frac{(v^q)^T \times v^k}{\sqrt{d}}\right)$

# Multi-head: 셀프 어텐션의 복회 수행을 통한 복수의 헤드 생성. 
# 
# N, s, d개의 텐서에 k개의 셀프 어텐션 벡터를 생성할 때, 헤드에 대한 차원 축을 생성하여 N, k, S, d/k 텐서 형태를 구성. 해당 텐서는 k개의 셀프 어텐션된 N, S, d/k 텐서를 의미

# 순방향 신경망은 선형 임베딩과 ReLU로 이루어진 인공신경망, 혹은 1차원 합성곱이 주로 사용.
# 
# TF 인코더는 복수의 TF encoder block으로 구성. 이전 블록에서ㅓ 출력된 벡터는 다음 블록의 입력으로 전달되어, 인코더 블록을 통과하며 점차 입력 시퀀스의 정보가 추상화.
# 
# 인코더 블록에서 출력된 벡터는 디코더에서 사용. 디코더의 MHA 모듈에서 참조되는 키, 값, 벡터로 활용.

# ## TF Decoder

# 위치 인코딩이 적용된 target data의 입력 임베딩을 입력받고, 디코더에 위치정보를 추가함으로써 디코더가 입력 시퀀스의 순서 정보를 학습할 수 있게끔 함.
# 
# 인코더의 MHA 모듈은 Casuality를 반영한 Masked MHA 모듈로 대체(이하 MMHA). MMHA 모듈은 인코더의 MHA 모델과 유사하나, 어텐션 스코어 맵을 계산할 때 쿼리 벡터가 해당하는 순서의 키 벡터만을 바라볼 수 있도록 마스크를 씌움. 해당 마스크를 적용하면 self attention에서 현재 위치 이전의 단어들만 참조할 수 있게 되며, 인과성이 보장.
# 
# MMHA 어텐션 모듈에서는 마스크 영역에 -inf 마스크를 더함으로써 해당 영역의 attention score값을 0에 가깝게 만들어 줄 수 있음.
# 
# 해당 방식으로, MMHA 모듈은 인코더의 멀티 헤드 어텐션 모듈과 유사하나, 인과성을 보장하며 self attention을 수행할 수 있게 해 줌.

# decoder의 MHA에서는 타깃 데이터가 쿼리 벡터로 사용되며, 인코더의 소스 데이터가 키와 값 벡터로 사용. 
# 
# 따라서, 쿼리 벡터는 타깃 데이터의 위치 정보를 포함한 입력 임베딩과 위치 인코딩을 더한 벡터

# 쿼리, 키, 값 벡터를 이용하여 attention score map을 계산 후, softmax 함수를 적용해 어텐션 가중치를 구함. 최종적으로, attention 가중치와 값 벡터를 가중합해 MHA의 출력 벡터를 구함.

# In[10]:


import torchtext
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import spacy


# In[11]:


def generate_tokens(text_iter, language):
    language_index = {src_language: 0, tgt_language: 1}
    
    for text in text_iter:
        yield token_transform[language](text[language_index[language]])
        
    
# def generate_tokens(text_iter, language, tokenizer):
#     for text in text_iter:
#         yield [token.text for token in tokenizer(text)]

src_language = 'de'
tgt_language = 'en'

unk_idx, pad_idx, bos_idx, eos_idx = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

token_transform = {
    src_language: get_tokenizer('spacy', language='de_core_news_sm'),
    tgt_language: get_tokenizer('spacy', language='en_core_web_sm'),
}


print('token_transform:')
print(token_transform)

# dataset = load_dataset('multi30k', split='train')

vocab_transform = {}
for language in [src_language, tgt_language]:
    train_iter = Multi30k(split='train', language_pair = (src_language, tgt_language))
#     tokenizer = token_transform[language]
    
#     for example in dataset:
#         tokens = generate_tokens([example[language]], language, tokenizer)
#         counter.update(next(tokens))

#     vocab = {token: idx + len(special_symbols) for idx, (token, _) in enumerate(counter.items())}
#     for i, sym in enumerate(special_symbols):
#         vocab[sym] = i
#     vocab_transform[language] = vocab
    
    vocab_transform[language] = build_vocab_from_iterator(
        generate_tokens(train_iter, language),
        min_freq = 1,
        specials = special_symbols,
        special_first = True
    )


# In[12]:


for language in [src_language, tgt_language]:
    vocab_transform[language].set_default_index(0)

print('vocab_transform:')
print(vocab_transform)


# 독어 말뭉치(de_core_news_sm)와 영어 말뭉치(en_core_web_sm)에 대하여 각 토크나이저와 어휘사전을 생성. 
# 
# get_tokenizer 함수는 사용자가 지정한 토크나이저를 가져오는 유틸리티 함수. 사전 학습듼 모델을 가져옴. 해당 값을 token_transform 변수에 저장

# vocab_transform 변수는 토큰을 인덱스로 변환하는 함수를 저장. Multi30k 데이터셋을 활용하여 독어-영어의 튜플 형식으로 데이터를 로드.

# 불러온 데이터에서 build_vocab_from_iterator 함수와 generate_tokens 함수로 언어별 어휘 사전을 생성.

# build_vocab_from_iterator 함수는 생성된 토큰을 통해 단어 집합을 생성. 최소 빈도는 토큰화된 단어들의 최소 빈도수를 지정.

# set_default_index method는 인덱스의 기본값을 설정하므로, 어휘 사전에 없는 토큰인 unk의 인덱스를 할당

# In[13]:


import torch
import math
from torch import nn

class positional_encoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model)
        )
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position*div_term)
        pe[:, 0, 1::2] = torch.cos(position*div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x+self.pe[:x.size(0)]
        return self.dropout(x)


# In[14]:


class token_embedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
        
    def forward(self, tokens):
        return self.embedding(tokens.long())*math.sqrt(self.emb_size)


# In[15]:


class seq2seq_transformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        emb_size,
        max_len,
        nhead,
        src_vocab_size,
        tgt_vocab_size,
        dim_feedforward,
        dropout = 0.1,
    ):
        super().__init__()
        self.src_tok_emb = token_embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = token_embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = positional_encoding(
            d_model = emb_size, max_len = max_len, dropout = dropout
        )
        self.transformer = nn.Transformer(
            d_model = emb_size,
            nhead = nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward = dim_feedforward,
            dropout = dropout
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        
    def forward(
        self,
        src,
        trg,
        src_mask,
        tgt_mask,
        src_padding_mask,
        tgt_padding_mask,
        memory_key_padding_mask,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src = src_emb,
            tgt = tgt_emb,
            src_mask = src_mask,
            tgt_mask = tgt_mask,
            memory_mask = None,
            src_key_padding_mask = src_padding_mask,
            tgt_key_padding_mask = tgt_padding_mask,
            memory_key_padding_mask = memory_key_padding_mask
        )
        return self.generator(outs)
    
    def encode(self, src, src_mask):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )
    
    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )


# ### transformer class

# In[16]:


transformer = torch.nn.Transformer(
    d_model = 512, # 임베딩 차원, 모델의 입력과 출력 차원의 크기. 임베딩 차원 크기와 동일.
    nhead = 8, # MHA 헤드 개수 정의. 모델이 어텐션을 수행하는 방법에 관여. 헤드가 많을수록 병렬 처리 능력이 증가하나, 그만큼 모댈 매개변수도 증가.
    num_encoder_layers = 6, # 인코더의 계층 수. 계층 개수가 많을수록 복잡한 문제를 해결하나, 과적합 문제도 가능
    num_decoder_layers = 6, # 디코더의 계층 수, 계층 개수가 많을수록 복잡한 문제를 해결하나, 과적합 문제도 가능
    dim_feedforward=2048, # 순방 신경망 크기. 순방향 신경망의 은닉층 크기를 정의. tf 계층 각 입력 위치에 독립적으로 적용. 모델의 복잡도와 성능에 관여.
    dropout = 0.1, 
    activation = torch.nn.functional.relu, 
    layer_norm_eps = 0.0001, # 계층 정규화 입실론. 계층 정규화를 수행할 때 분모에 더해지는 입실론 값을 정의
)


# ```
# transformer = torch.nn.Transformer(
#     d_model = 512, # 임베딩 차원, 모델의 입력과 출력 차원의 크기. 임베딩 차원 크기와 동일.
#     nhead = 8, # MHA 헤드 개수 정의. 모델이 어텐션을 수행하는 방법에 관여. 헤드가 많을수록 병렬 처리 능력이 증가하나, 그만큼 모댈 매개변수도 증가.
#     num_encoder_layers = 6, # 인코더의 계층 수. 계층 개수가 많을수록 복잡한 문제를 해결하나, 과적합 문제도 가능
#     num_decoder_layers = 6, # 디코더의 계층 수, 계층 개수가 많을수록 복잡한 문제를 해결하나, 과적합 문제도 가능
#     dim_feedforward=2048, # 순방 신경망 크기. 순방향 신경망의 은닉층 크기를 정의. tf 계층 각 입력 위치에 독립적으로 적용. 모델의 복잡도와 성능에 관여.
#     dropout = 0.1, 
#     activation = torch.nn.functional.relu, 
#     layer_norm_eps = 1e-0.5 # 계층 정규화 입실론. 계층 정규화를 수행할 때 분모에 더해지는 입실론 값을 정의
# )
# ```

# ### 트랜스포머 순방향 메소드

# ```
# output = transformer.forward(
#     src, # 인코더 시퀀스. 소스 시퀀스 길이, 배치 크기, 임베딩 차원 형태의 데이터 인풋
#     tgt, # 디코더 시권스. 타깃 시퀀스 길이, 배치 크기, 임베딩 차원 형태의 데이터 인풋 
#     src_mask = None, # 소스 시퀀스의 마스크. 소스 시퀀스 길이, 시퀀스 길이 형태의 데이터 인풋. 
#                      # 0인 경우, 해당 위치에서 모든 입력 단어가 동일한 가중치를 갖고 어텐션이 수행됨. 
#                      # 1이라면, 모든 입력 단어의 가중치가 0. 어텐션 연산이 수행되지 아니함.
#     tgt_mask = None, # 타깃 시퀀스의 마스크. 타깃 시퀀스 길이, 시퀀스 길이 형태의 데이터 인풋
#                      # 0인 경우, 해당 위치에서 모든 입력 단어가 동일한 가중치를 갖고 어텐션이 수행됨. 
#                      # 1이라면, 모든 입력 단어의 가중치가 0. 어텐션 연산이 수행되지 아니함.
#     memory_mask = None, # 
#     src_key_padding_mask = None,
#     tgt_key_padding_maks = None,
#     memory_key_padding_mask = None,
# )
# ```

# In[17]:


from torch import optim
batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[18]:


model = seq2seq_transformer(
    num_encoder_layers= 3,
    num_decoder_layers= 3,
    emb_size = 512,
    max_len = 512,
    nhead = 8,
    src_vocab_size= len(vocab_transform[src_language]),
    tgt_vocab_size= len(vocab_transform[tgt_language]),
    dim_feedforward= 512
).to(device)


# In[19]:


criterion = nn.CrossEntropyLoss(ignore_index = pad_idx).to(device)
optimizer = optim.Adam(model.parameters())


# In[20]:


for main_name, main_module in model.named_children():
    print(main_name)
    for sub_name, sub_module in main_module.named_children():
        print("| ", sub_name)
        for ssub_name, ssub_module in sub_module.named_children():
            print("| ", ssub_name)
            for sssub_name, sssub_module in ssub_module.named_children():
                print("| |", sssub_name)


# batch data creation

# In[21]:


from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


# In[22]:


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func
    
def input_transform(token_ids):
    return torch.cat(
        (torch.tensor([bos_idx]), torch.tensor(token_ids), torch.tensor([eos_idx]))
    )
    
def collator(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[src_language](src_sample.rstrip('\n')))
        tgt_batch.append(text_transform[tgt_language](tgt_sample.rstrip('\n')))
        
    src_batch = pad_sequence(src_batch, padding_value = pad_idx)
    tgt_batch = pad_sequence(tgt_batch, padding_value = pad_idx)
    return src_batch, tgt_batch


# In[23]:


text_transform = {}


# In[24]:


for language in [src_language, tgt_language]:
    text_transform[language] = sequential_transforms(
        token_transform[language], vocab_transform[language], input_transform
    )
    
data_iter = Multi30k(split = 'valid', language_pair = (src_language, tgt_language))
dataloader = DataLoader(data_iter, batch_size = batch_size, collate_fn = collator)
source_tensor, target_tensor = next(iter(dataloader))


# In[25]:


print('(source, target):')
print(next(iter(data_iter)))


# In[26]:


print('source_batch:', source_tensor.shape)
print(source_tensor)


# In[27]:


print('target_batch:', target_tensor.shape)
print(target_tensor)


# ### attention mask

# In[28]:


def generate_squaree_subsequent_mask(s):
    mask = (torch.triu(torch.ones((s, s), device=device))==1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float('-inf'))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    
    tgt_mask = generate_squaree_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    
    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# In[29]:


target_input = target_tensor[:-1, :]
target_out = target_tensor[1:, :]


# In[30]:


source_mask, target_mask, source_padding_mask, target_padding_mask = create_mask(
    source_tensor, target_input
)


# In[31]:


print('source:', source_mask.shape)
print(source_mask)


# In[32]:


print(target_mask)


# In[33]:


def run(model, optimizer, critierion, split):
    model.train() if split == 'train' else model.eval()
    data_iter = Multi30k(split=split, language_pair = (src_language, tgt_language))
    dataloader = DataLoader(data_iter, batch_size = batch_size, collate_fn = collator)
    
    losses = 0
    for source_batch, target_batch in dataloader:
        source_batch = source_batch.to(device)
        target_batch = target_batch.to(device)
        
        target_input = target_batch[:-1, :]
        target_output = target_batch[1:, :]
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            source_batch, target_input
        )
        
        logits = model(
            src = source_batch,
            trg = target_input,
            src_mask = src_mask,
            tgt_mask = tgt_mask,
            src_padding_mask = src_padding_mask,
            tgt_padding_mask = tgt_padding_mask,
            memory_key_padding_mask = src_padding_mask
        )
        
        optimizer.zero_grad()
        loss = criterion(logits.reshape(-1, logits.shape[-1]), target_output.reshape(-1))
        if split == 'train':
            loss.backward()
            optimizer.step()
        losses += loss.item()
        
    return losses / len(list(dataloader))


# In[34]:


for epoch in range(5):
    train_loss = run(model, optimizer, criterion, 'train')
    val_loss = run(model, optimizer, criterion, 'valid')
    print(f'Epoch: {epoch+1}, train_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}')


# In[35]:


def greedy_decode(model, source_tensor, source_mask, max_len, start_symbol):
    source_tensor = source_tensor.to(device)
    source_mask = source_mask.to(device)
    
    memory = model.encode(source_tensor, source_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    
    for i in range(max_len - 1):
        memory = memory.to(device)
        target_mask = generate_squaree_subsequent_mask(ys.size(0))
        target_mask = target_mask.type(torch.bool).to(device)
        
        out = model.decode(ys, memory, target_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(source_tensor.data).fill_(next_word)], dim=0
        )
        if next_word == eos_idx:
            break
    
    return ys


# In[36]:


def translate(model, source_sentence):
    model.eval()
    source_tensor = text_transform[src_language](source_sentence).view(-1, 1)
    num_tokens = source_tensor.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, source_tensor, src_mask, max_len = num_tokens+5, start_symbol = bos_idx).flatten()
    output = vocab_transform[tgt_language].lookup_tokens(list(tgt_tokens.cpu().numpy()))[1:-1]
    return ' '.join(output)


# In[37]:


output_oov = translate(model, 'Eine Gruppe von Menschen steht vor einem Iglu')
output = translate(model, 'Eine Gruppe von Menschen steht vor einem Gebäude')


# In[38]:


print(output_oov)


# In[39]:


print(output)


# ## GPT

# transformer의 디코더를 복층화한 언어 모델. 
# 
# 자연어 생성과 같은 언어 모델링 작업에서 높은 성능. 

# ## GPT II

# In[40]:


from transformers import GPT2LMHeadModel


# GPT2LMHeadModel
# 
# GPT2: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
# 
# GPT3: OpenAI API를 통해 사용
# 
# GPT-Neo: 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'ElutherAI/gpt-neo-2.7B'
# 
# GPT-J: 'EleutherAI/gpt-j-6B'
# 
# BERT: 'bert-base-uncased', 'bert-large-uncased'
# 
# T5: 't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'
# 
# RoBERTa: 'roberta-base', 'roberta-large'
# 
# DistilGPT2: 'distilgpt2'

# transformers library에서 사용 가능한 model 클래스 대표
# 
# BERT
# ```
# from transformers import BertModel, BertForSequenceClassification, BertForTokenClassification
# model = BertModel.from_pretrained('bert-base-uncased')
# ```
# 
# RoBERTa(BERT 개선)
# ```
# from transformers import RobertaModel, RobertaForSequenceClassification, RobertaForTokenClassification
# model = RobertaModel.from_pretrained('roberta-base')
# ```
# 
# DistilBERT
# ```
# from transformers import DistilBertModel, DistilBertForSequenceClassification, DistilBertForTokenClassification
# model = DistilBertModel.from_pretrained('distilbert-base-uncased')
# ```
# 
# T5(텍스트 변환 / 번역, 요약)
# ```
# from transformers import T5ForConditionalGeneration
# model = T5ForConditionalGeneration.from_pretrained('t5-base')
# ```
# 
# GPT-Neo
# ```
# from transformers import GPTNeoForCausalLM
# model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')
# ```
# 
# GPT-J(대용량 GPT 모델)
# ```
# from transformers import GPTJForCausalLM
# model = GPTJForCausalLM.from_pretrained('EleutherAI/gpt-j-6B')
# ```
# 
# BART(텍스트 생성과 요약)
# ```
# from transformers import BartForConditionalGeneration
# model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
# ```
# 
# XLNet(tf-XL)
# ```
# from transformers import XLNetModel, XLNetForSequenceClassification
# model = XLNetModel.from_pretrained('xlnet-base-cased')
# ```
# 
# Albert(경량 BERT)
# ```
# from transformers import AlbertModel, AlbertForSequenceClassification
# model = AlbertModel.from_pretrained('albert-base-v2')
# ```
# 
# Longformer(대용량 문서 처리)
# ```
# from transformers import LongformerModel, LongformerForSequenceClassification
# model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
# ```

# In[41]:


model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path= 'gpt2')


# In[42]:


for main_name, main_module in model.named_children():
    print(main_name)
    for sub_name, sub_module in main_module.named_children():
        print("| ", sub_name)
        for ssub_name, ssub_module in sub_module.named_children():
            print("| ", ssub_name)
            for sssub_name, sssub_module in ssub_module.named_children():
                print("| |", sssub_name)


# In[43]:


from transformers import pipeline

generator = pipeline(task = 'text-generation', model='gpt2')
outputs = generator(
    text_inputs = 'Machine learning is',
    max_length = 20,
    num_return_sequences = 5,
    pad_token_id = generator.tokenizer.eos_token_id
)


# In[44]:


print(outputs)


# In[45]:


import torch
from torchtext.datasets import CoLA
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


# In[46]:


from datasets import load_dataset


# In[47]:


dataset = load_dataset('glue', 'cola')
print(dataset)


# In[48]:


def collator(batch, tokenizer, device):
    source, labels, texts = zip(*batch)
    tokenized = tokenizer(
        texts,
        padding = 'longest',
        truncation = True,
        return_tensors = 'pt'
    )
    
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    return input_ids, attention_mask, labels

train_data = list(CoLA(split='train'))
valid_data = list(CoLA(split='dev'))
test_data = list(CoLA(split='test'))
# 버전 호환성 에러로 load dataset으로 대체
# 해결됨

# train_data = dataset['train']
# valid_data = dataset['validation']
# test_data = dataset['test']


# In[49]:


tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


# In[50]:


epochs = 3
batch_size=16
device = 'cuda'


# In[51]:


train_dataloader = DataLoader(
    train_data,
    batch_size = batch_size,
    collate_fn = lambda x: collator(x, tokenizer, device),
    shuffle = True
)


# In[52]:


valid_dataloader = DataLoader(
    valid_data, batch_size=batch_size, collate_fn = lambda x: collator(x, tokenizer, device)
)
test_dataloader = DataLoader(
    test_data, batch_size = batch_size, collate_fn = lambda x: collator(x, tokenizer, device)
)


# In[53]:


print('train_length: ', len(train_data))
print('valid_length: ', len(valid_data))
print('test_length: ', len(test_data))


# CoLA dataset
# 
# train / dev(validation) / test
# 
# 특수 토큰 중, pad 토큰이 없으므로, 문장 분류 모델의 학습을 위해 eos 토큰으로 패딩 토큰을 대체.
# 
# collator를 통해 배치를 토크나이저로 토큰화, padding, truncation, return_tensors 작업을 수행
# 
# padding 인자를 longest로 설정하면, 가장 긴 시퀀스에 대해 패딩을 적용, 절사에 인자를 True로 설정하면, 입력 시퀀스 길ㅇ리가 최대 길이를 초과하는 경우 해당 시퀀스를 절사.
# 
# 반환 형식 설정에 pt를 입력하면, pytorch tensor 형태로 결과를 반환.
# 
# tokenizer는 토큰 id(input_ids)와 어텐션 마스크(attention mask)를 반환. 

# In[54]:


from torch import optim
from transformers import GPT2ForSequenceClassification


# In[55]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[56]:


model = GPT2ForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path = 'gpt2',
    num_labels = 2
).to(device)
model.config.pad_token_id = model.config.eos_token_id
optimizer = optim.Adam(model.parameters(), lr=5e-5)


# In[57]:


import numpy as np
from torch import nn


# In[58]:


def calc_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[59]:


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


# In[60]:


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


# In[61]:


best_loss = 10000
for epoch in range(epochs):
    train_loss = train(model, optimizer, train_dataloader)
    val_loss, val_accuracy = evaluation(model, valid_dataloader)
    print(f'Epoch: {epoch+1} train loss: {train_loss:.4f} val loss: {val_loss:.4f} val_acc: {val_accuracy:.4f}')
    
    if val_loss < best_loss:
        best_loss = val_loss


# 문장 분류 모델의 정의와 학습

# In[64]:


from torch import optim
from transformers import GPT2ForSequenceClassification

model = GPT2ForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path = 'gpt2',
    num_labels = 2
).to(device)


# In[65]:


model.config.pad_token_id = model.config.eos_token_id
optimizer = optim.Adam(model.parameters(), lr=5e-5)


# In[67]:


torch.save(model.state_dict(), './GPT2_for_sequence_classification.pt')


# In[68]:


test_loss, test_ac = evaluation(model, test_dataloader)


# In[69]:


print(f'test loss: {test_loss:.4f}')
print(f'test_ac: {test_ac:.4f}')


# ## BERT
# 
# (Bidirectional Encoder Representations from Transformers)

# 입력 시퀀스를 양측에서 처리하여 이전과 이후의 단어를 모두 참조하며 단어의 의미를 파악.
# 
# 기존 모델들보다 정확하게 문맥을 파악하고 다양한 자연어 처리 작업에서 높은 성능을 보임.
# 
# 대규모 데이터를 통해 사전 학습되어 있으므로 전이 학습에 주로 활용. 
# 
# BERT는 일부나 전체를 다른 작업에서 재사용하여 적은 데이터로도 높은 정확도를 달성. 

# ### Masked Langauge Modeling, MLM

# 입력 문장에서 임의로 일부 단어를 마스킹, 해당 단어를 예측.

# ### Next Sentence Prediction, NSP

# 두 문장이 주어졌을 때, 두 번째 문장이 첫 번째 문장의 다음에 오는 문장인지의 여부 판단.

# BERT 모델의 토큰
# 
# CLS / SEP / MASK

# CLS:
# 
# 입력 문장의 시작. 해당 토큰을 사용해 문장 분류 작업을 위한 정보를 제공.
# 
# 모델은 문장이 어떤 유형의 것인지 미리 파악 가능하게 됨. 
# 
# 
# SEP:
# 
# 입력 문장 내에서 문장의 구분을 위해 사용. 문장 분률 작업에서 복수 문장을 입력으로 받을 때, SEP 토큰을 통해 문장을 구분.
# 
# 
# MASK:
# 
# 입력 문장 내에서 임의로 선택된 단어를 가리키는 토큰. 주어진 문장에서 단어를 가린 후 모델의 학습과 예측에 활용. 

# In[ ]:





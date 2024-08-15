#!/usr/bin/env python
# coding: utf-8

# ## BART
# 
# Bidirectional Auto-Regressive Transformer

# Transformer based model
# 
# BERT의 Encoder와 GPT의 디코더의 결합. Sequence to Sequence, Sep2sep 구조로 Denoising Autoencoder로 사전 학습.

# BERT 인코더는 입력 문장에서 일부 단어를 무작위로 마스킹해 처리, 마스킹된 단어를 맞추게 학습. BERT 인코더는 문장 전체 맥락을 이해하고 문맥 내 단어 간 상호작용을 파악.
# 
# GPT는 언어 모델을 통해 문장의 이전 토큰을 입력으로 받고 다음에 올 토큰을 맞추도록 학습. 이를 통해 GPT는 문장 내 단어들의 순서와 문맥을 파악. 다음에 올 단어를 예측.

# BART는 사전 학습 시 노이즈 제거 오토인코더를 사용하므로, 입력 문장에 임의의 노이즈를 추가하고 원래 문장을 복원하도록 학습.
# 
# 노이즈가 추가된 텍스트를 인코더에 입력하고 원본 텍스트를 디코더에 입력해 디코더가 원본 텍스트를 생성 가능하게 학습하는 방식.
# 
# 이에 BART는 문장 구조와 의미를 보존하며 다양한 변형을 학습 가능함. 입력 문장에 제약 없이 노이즈 기법을 적용 가능하므로 더 풍부한 언어적 지식을 습득 가능.

# 인코더를 사용하여 순방향 정보만 인식 가능한 GPT의 단점을 개선하여 양방향 문맥 정보를 반영, 디코더를 사용함으로써 문장 생성 분야에서 뛰어나지 않았던 BERT의 단점을 개선.

# BERT vs BART
# 
# BERT는 인코더만 사용. BART는 인코더와 디코더를 모두 사용. 
# 
# BART는 인코더와 디코더를 모두 사용하므로 트랜스포머와 유사한 구조를 가지나, tf에서는 인코더의 모든 계층과 디코더의 모든 계층 사이의 어텐션 연산을 수행한다면, BART는 인코더의 마지막 계층과 디코더의 각 계층 사이에서만 어텐션 연산을 수행.

# BART에서는 인코더의 마지막 계층과 디코더의 각 계층 사이에서만 어텐션 연산을 수행하므로, 정보 전달을 최소화하고 메모리 사용량을 줄일 수 있음.

# BART encoder에서는 입력 문장의 각 단어를 임베딩하고 복층 인코더를 거쳐 마지막 계층에서는 입력 문장 전체의 의미를 가장 잘 반영하는 벡터가 생성.
# 
# 이렇게 생성된 벡터는 디코더가 출력 문장을 생성할 때 참고됨. 디코더의 각 계층에서는 이전 계층에서 생성된 출력 문장의 정보를 활용하여 출력 문장을 생성.

# ### 사전학습의 방법

# BART encoder는 Bert의 마스킹된 언어 모델(MLM) 외에도 다양하게 노이즈 기법을 사용.
# 
# 토큰 마스킹 기법 외에도 토큰 삭제, 문장 교환, 문서 회전, 텍스트 채우기 등.

# ### 미세 조정의 방법

# BART는 인코더와 디코더를 모두 사용. 미세 조정 시 각 다운스트림 작업에 맞게 입력 문장을 구성.
# 
# 즉, 인코더와 디코더에 다른 문장 구조로 입력.

# BART는 tf 디코더를 사용하기 때문에 BERT가 해결하지 못했던 문장 생성 작업을 수행 가능.
# 
# 특히, 입력값을 조작하여 출력을 생성하는 추상적 질의응답(Abstractive Question Answering)과 문장 요약(Summarisation) 등의 작업에 적합.

# BART는 이를 통해 문장의 의미를 파악하고, 새로운 문장을 생성하는 능력을 확보.

# In[1]:


import sys
sys.path.append("C:/Users/dohyeong/miniconda3/Lib/site-packages/")


# In[2]:


import numpy as np
from datasets import load_dataset
import torch
from transformers import BartTokenizer
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence

from torch import optim
from transformers import BartForConditionalGeneration


# In[3]:


news = load_dataset('argilla/news-summary', split='test')
df = news.to_pandas().sample(5000, random_state=42)[['text', 'prediction']]
df['prediction'] = df['prediction'].map(lambda x: x[0]['text'])
train, valid, test = np.split(
    df.sample(frac=1, random_state=42), [int(0.6*len(df)), int(0.8*len(df))]
)


# In[4]:


print(f'source news: {train.text.iloc[0][:2000]}')


# In[5]:


len(train)


# In[6]:


def make_dataset(data, tokenizer, device):
    tokenized = tokenizer(
        text = data.text.tolist(),
        padding = 'longest',
        truncation = True,
        return_tensors = 'pt'
    )
    labels = []
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    for target in data.prediction:
        labels.append(tokenizer.encode(target, return_tensors = 'pt').squeeze())
    labels = pad_sequence(labels, batch_first = True, padding_value = -100).to(device)
    return TensorDataset(input_ids, attention_mask, labels)


# In[7]:


def get_dataloader(dataset, sampler, batch_size):
    data_sampler = sampler(dataset)
    dataloader = DataLoader(dataset, sampler = data_sampler, batch_size = batch_size)
    return dataloader


# In[8]:


epochs = 3
batch_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# cuda out of memory 에러로 batch_size 축소


# In[9]:


device


# In[11]:


news.shape


# In[12]:


tokenizer = BartTokenizer.from_pretrained(
    pretrained_model_name_or_path = 'facebook/bart-base'
)


# In[13]:


train_dataset = make_dataset(train, tokenizer, device)
train_dataloader = get_dataloader(train_dataset, RandomSampler, batch_size)

valid_dataset = make_dataset(valid, tokenizer, device)
valid_dataloader = get_dataloader(valid_dataset, RandomSampler, batch_size)

test_dataset = make_dataset(test, tokenizer, device)
test_dataloader = get_dataloader(test_dataset, RandomSampler, batch_size)


# In[14]:


model = BartForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path= 'facebook/bart-base'
).to(device)


# In[15]:


optimizer = optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)


# ### ROUGE(Recall-Oriented Understudy for Gisting Evaluation, ROUGT) score
# 
# 생성된 요약문과 정답 요약문이 얼마나 유사한지를 평가하기 위해 토큰의 N-gram 정밀도의 재현율을 이용해 평가하는 지표.

# #### ROUGE-L / ROUGE-LSUM / ROUGE-W
# 
# ROUGE-L: 최장 공통부분 수열(Longest Common Subsequence, LCS) 기반의 통계
# 
# ROUGE-LSUM: ROUGE-L의 변형. 텍스트 내의 개행 문자를 문장 경계로 인식, 각 문장쌍에 대해 LCS를 계산 후, union-LCS 값을 계산(각 문장 쌍의 LCS의 합집합).
# 
# ROUGE-W: 가중치가 적용된 LCS. 연속된 LCS에 가중치를 부여하여 계산. 

# huggingface의 평가(evaluate) 라이브러리를 통해 루지 점수를 계산 가능(+ rouge_score 라이브러리 / Abseil 라이브러리)

# In[16]:


import numpy as np
import evaluate


# In[17]:


def calc_rouge(preds, labels):
    preds = preds.argmax(axis = -1)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens = True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens = True)
    
    rouge2 = rouge_score.compute(
        predictions = decoded_preds,
        references = decoded_labels
    )
    return rouge2['rouge2']


# In[23]:


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
    print('trained')
    return train_loss


# In[24]:


def evaluation(model, dataloader):
    with torch.no_grad():
        model.eval()
        val_loss, val_rouge = 0.0, 0.0
        
        for input_ids, attention_mask, labels in dataloader:
            outputs = model(
                input_ids = input_ids, attention_mask = attention_mask, labels = labels
            )
            logits = outputs.logits
            loss = outputs.loss
            
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            rouge = calc_rouge(logits, label_ids)
            
            val_loss += loss
            val_rouge += rouge
            
    val_loss = val_loss / len(dataloader)
    val_rouge = val_rouge / len(dataloader)
    print('evaluated')
    return val_loss, val_rouge
    
rouge_score = evaluate.load('rouge', tokenizer=tokenizer)


# In[35]:


from datetime import datetime


# In[36]:


now = datetime.now()


# In[38]:


now.strftime('%H:%M:%S')


# In[39]:


datetime.now().strftime("%H:%M:%S")


# In[40]:


best_loss = 1000
for epoch in range(epochs):
    print('start:', datetime.now().strftime("%H:%M:%S"))
    train_loss = train(model, optimizer, train_dataloader)
    val_loss, val_accuracy = evaluation(model, valid_dataloader)
    print(f'epoch: {epoch + 1} train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} val_ac: {val_accuracy:.4f}')
    print('done:', datetime.now().strftime("%H:%M:%S"))
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), './bart_model.pt')
        print('model saved')


# In[41]:


device


# In[42]:


test_loss, test_rouge_score = evaluation(model, test_dataloader)


# In[43]:


print(f'test_loss: {test_loss:.4f}')
print(f'test_rouge2_score: {test_rouge_score:.4f}')


# In[44]:


from transformers import pipeline


# In[45]:


summarizer = pipeline(
    task = 'summarization',
    model = model,
    tokenizer = tokenizer,
    max_length = 54,
    device = device
)


# In[46]:


for index in range(5):
    news_text = test.text.iloc[index]
    summarisation = test.prediction.iloc[index]
    predicted_summarisation = summarizer(news_text)[0]['summary_text']
    print(f'summary: {summarisation}')
    print(f'model: {predicted_summarisation}')


# In[ ]:





# In[ ]:





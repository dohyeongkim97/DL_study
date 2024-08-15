#!/usr/bin/env python
# coding: utf-8

# # Chapter IV

# Data Collection
# 
# Feature Engineering
# 
# Model Transform
# 
# Early Stopping
# 
# Batch Normalisation
# 
# Weight Initialisation
# 
# Regulation

# ## Batch Normalisation

# ICT(internal covariate shift)를 줄여 과대적합을 방지. CNN or FeedForward Neural Network
# 
# Normalise by the average and variation of each mini batches

# Layer Normalisation, Match Normalisation, Instance Normalisation, Group Normalisation

# In[2]:


import torch
from torch import nn


# In[ ]:


# 모델이 작고 변동성이 거의 없는 경우. 가중치 초기화 매소드를 기본적으로 적용


# In[3]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(2, 1)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.layer[0].weight)
        self.layer[0].bias.data.fill_(0.01)
        
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.01)

model = Net()


# In[ ]:


# 모델이 크고 변동성이 큰 경우. 가중치 초기화 메소드를 모듈화하여 적용.


# In[4]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        
        self.fc = nn.Linear(2, 1)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.01)
        print(f'apply: {module}')
        
model = Net()


# In[6]:


## regulation : 과대적합 문제를 방지하기 위해 사용. 손실함수에 규제(penalty)를 가하는 방식. 일반화성능의 향상 목적. 분산을 낮춰 각 값들의 차이점에 덜 민감하게 만드는 것이 목적.


# ### L1

# ```
# for x, y in train_dataloader:
#     output = model(x)
#     
#     _lambda = 0.5
#     l1_loss = sum(p.abs().sum() for p in model.parameters())
#     
#     loss = criterion(output, y) + _lambda * l1_loss
# ```

# 모델 가중치 절댓값의 합 사용. 모델의 가중치를 모두 계산하여 모델을 갱신해야 하므로 계산복잡도(Computational Complexity) 증가. 
# 
# L1정칙화는 미분이 불가능하므로 역전파를 계산하는 데 더 큰 리소스를 소모. 람다값이 적절하지 않으면 가중치 값이 너무 작아져 모델을 해석하기 어렵게 만들 수 있음.
# 
# 복수 회차 반복하여 최적의 람다값을 탐사. 
# 
# 주로 선형 모델에 적용됨. 선형회귀 모델에 L1정칙화를 적용하는 것을 Lasso(Least Absolute Shrinkage and Selection Operator) 회귀라고 칭함.

# ### L2

# L2 norm을 사용한 규제. 벡터나 행렬 값의 크기를 계산. 손실함수에 가중치 제곱의 합을 추가하여 과대적합을 방지하도록 규정.
# 
# L1과 동일하게 모델에 규제를 가함. 하나의 특징이 너무 중요한 요소가 되지 아니하도록 규제를 가하는 것에 의미를 둠.
# 
# Ridge Regulation이라고 표현. 

# ```
# for x, y in train_dataloader:
#     output = model(x)
#     
#     _lambda = 0.5
#     l2_loss = sum(p.pow(2.0).sum() for p in model.parameters())
#     
#     loss = criterion(output, y) + _lambda * l2_loss
# ```

# ### Weight Decay

# 일반적으로 L2 regulation과 동의어. 광의의 의미로는 손실 함수에 규제 항을 추가하는 기술 자체.

# ### Momentum

# 경사 하강 알고리즘의 변형. 이전에 이동했던 방향과 기울기의 크기를 고려하여 가중치를 갱신. 지수 가중 이동평균을 사용. 이전 기울기의 일부를 현재 항에 추가하여 가중치를 갱신.

# ### Elastic Net

# L1 정규화와 L2 정규화를 결합하여 사용하는 방식. L1은 모델이 희박한 가중치를 갖도록, L2는 큰 가중치를 갖지 않도록 규제. 희소성과 작은 가중치의 균형을 찾기 위해 사용하는 방식.

# ### Drop Out

# 일부 노드를 제거하여 사용하는 방식. Voting 효과와 Model Averaging이 가능. 그러나 복수 회차를 통해 voting을 적용해야 하기에 훈련 시간은 늘어남.

# ```
# from torch import nn
# 
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Linear(10, 10)
#         self.dropout = nn.Dropout(p = 0.5)
#         self.layer2 = nn.Linear(10, 10)
#         
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.dropout(x)
#         x = self.layer2(x)
#         return x
# ```    
# 

# ### Gradient Clipping

# 모델 학습 시 기울기가 너무 커지는 것을 방지하는 기술. 가중치 최댓값을 규제하여 최대 임곗값을 초과하지 아니하도록 기울기를 잘라(Clipping) 설정한 임곗값으로 변경. 

# ```
# grad_norm = torch.nn.utils.clip_grad_norm_(
#     parameters,
#     max_norm,
#     norm_type = 2.0
# )
# ```

# ```
# for x, y in train_dataloader:
#     output = model(x)
#     loss = criterion(output, y)
#     
#     optimizer.zero_rad()
#     loss.backward()
#     
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
#     
#     optimizer.step()
#     
# ```

# # Text Data

# insert and delete

# insertion

# In[8]:


import nlpaug.augmenter.word as naw


# In[19]:


texts = [
    'Those who can imagine anything, can create the impossible.',
    'We can only see a short distance ahead, but we can see plenty there that needs to be done.',
    'If a machine is expected to be infallible, it cannot also be intelligent.'
]

aug = naw.ContextualWordEmbsAug(model_path = 'bert-base-uncased', action='insert')
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f'src: {text}')
    print(f"dist: {augmented}")
    print("______________")


# swap

# In[20]:


aug = naw.RandomWordAug(action = 'swap')
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f'src: {text}')
    print(f"dist: {augmented}")
    print("______________")


# replacement

# In[22]:


aug = naw.SynonymAug(aug_src = 'wordnet')
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f'src: {text}')
    print(f"dist: {augmented}")
    print("______________")


# delete

# In[12]:


import nlpaug.augmenter.char as nac


# In[14]:


aug = nac.RandomCharAug(action = 'delete')
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f'src: {text}')
    print(f"dist: {augmented}")
    print("______________")


# back_translation

# In[28]:


back_translation = naw.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-mt-en-fr',
#     to_model_name='Helsinki-NLP/opus-mt-fr-en'
)

augmented_text = back_translation.augment(texts)


# In[27]:


texts


# In[38]:


back_translation_resolve = naw.BackTranslationAug(
    from_model_name = 'Helsinki-NLP/opus-mt-fr-en'
)

augmented_reverse = back_translation_resolve(augmented_Text)


# In[ ]:


augmented_reverse


# In[39]:


# First, create the back translation augmenter from English to French
back_translation_en_fr = naw.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-mt-en-fr',
    to_model_name='Helsinki-NLP/opus-mt-fr-en'
)

# Augment the texts
augmented_texts = back_translation_en_fr.augment(texts)
print("Augmented Texts:", augmented_texts)

# Then, create the back translation augmenter from French to English
back_translation_fr_en = naw.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-mt-fr-en',
    to_model_name='Helsinki-NLP/opus-mt-en-fr'
)

# Augment the already augmented texts back to English
back_translated_texts = back_translation_fr_en.augment(augmented_texts)
print("Back Translated Texts:", back_translated_texts)


# # Image Data

# library : torchvision, imgaug

# In[42]:


from PIL import Image
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]
)

image = Image.open('./images/cat.jpg')
transformed_image = transform(image)

print(transformed_image.shape)


# In[45]:


transformed_image[0][0]


# rotation

# In[47]:


transform = transforms.Compose(
    [
        transforms.RandomRotation(degrees = 30, expand=False, center=None),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
    ]
)


# cut and padding

# In[49]:


transform = transforms.Compose(
    [
        transforms.RandomCrop(size=(512, 512)),
        transforms.Pad(padding=50, fill=(127, 127, 255), padding_mode='constant')
    ]
)


# resize

# In[51]:


transform = transforms.Compose(
    [
        transforms.Resize(size=(512, 512))
    ]
)


# colour transformation

# In[53]:


transform = transforms.Compose(
    [
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.3, hue=0.3
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        ),
        transforms.ToPILImage()
    ]
)


# noise

# In[54]:


import numpy as np
from imgaug import augmenters as iaa


# In[55]:


class IaaTransforms:
    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.SaltAndPepper(p=(0.03, 0.07)),
            iaa.Rain(speed=(0.3, 0.7))
        ])
        
    def __call__(self, images):
        iamges = np.array(images)
        augmented = self.seq.augment_image(images)
        return Image.fromarray(augmented)
    
transform = transforms.Compose([
    IaaTransforms()
])


# Cutout and Random Erasing

# In[56]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomErasing(p=1.0, value=0),
    transforms.RandomErasing(p=1.0, value='random'),
    transforms.ToPILImage()
])


# Mixup and CutMix

# Mixup

# In[57]:


class Mixup:
    def __init__(self, target, scale, alpha=0.5, beta=0.5):
        self.target = target
        self.scale = scale
        self.alpha = alpha
        self.beta = beta
        
    def __call__(self, image):
        image = np.array(image)
        target = self.target.resize(self.scale)
        target = np.array(target)
        mix_image = image*self.alpha+target*self.beta
        
transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        Mixup(
            target = Image.open('./images/dog.jpg'),
            scale = (512, 512),
            alpha=0.5,
            beta=0.5
        )
    ]
)


# PreTrained Model

# transfer learning and backbone networks

# Backbone

# A model or a part of it that extract features from input data and give it to final classifier.
# 
# mentioned from VGG(Very Deep Convolutional Networks for Large Scale Image Recognition), ResNet(Deep Residual Learning for Image Recognition), Mask R-CNN

# Hyper-scale deep learning models like BERT, GPT, VGG-16, ResNet

# Transfer Learning

# Re-Use some pre-trained model to improve efficiency of some domains

# Dog-Cat model to Wolf-Lion model.

# In[ ]:





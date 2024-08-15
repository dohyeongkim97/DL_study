#!/usr/bin/env python
# coding: utf-8

# # CNN

# Convolution Neural Network. A sort of ANN which is mainly used for computer vision field like Image Recognition etc.
# 
# Specialised for extracting local features of input data. 

# Also can be used for NLP

# ## Filter

# aka Kernel or Window

# ## Padding

# adding fixed values to the edges of data. 

# ## Stride

# the size which the filter moves.
# 
# if stride be bigger, the output size be smaller.

# ## Channel

# when the input and filter composed by 3rd demension, channel make the values be calculated on same level. 
# 
# can expand the extracted feature while maintaning loc info of input data.

# usually set by conv-level, and it's different by the structure or object of the model

# in case that there are many output channels, each of channel can get each other features by input data. therefore model can get more of treats of features. so that it can solve some difficult problemes

# ## Dilation

# put terms into the filter and input data. usually 1.

# CNN layer classes

# ```
# conv = torch.nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size, ( filter size )
#     stride=1, ( move )
#     padding=0,
#     dilation=1,
#     groups=1, ( when group == 1, in channels and out channels bounded in one group. if more, group devided by several smaller groups, and conv)
#     bias=True,
#     padding_mode='zeros'
# )
# ```

# $L_{\text{out}} = \left[ \frac{L_{\text{in}} + 2 \times \text{padding} - \text{dilation} \times (\text{kernel} - 1) - 1}{\text{stride}} + 1 \right]$

# ## Activation Map

# a specific value gained adopting activate fucntion to specific feature map in CNN layers

# by several layer of activation map, NN can get abstract features of input images.

# ## Pooling

# feature map size reducing. compressing input data info and reduce calc power

# max pooling vs avg pooling

# ``` 
# pool = torch.nn.MaxPool2d(
#     kernel_size,
#     stride=None,
#     padding=0,
#     dilation=1
# )
# ```

# ```
# pool = ttorch.nn.AvgPool2d(
#     kernel_size,
#     stride=None,
#     padding=0,
#     count_include_pad=True
# )
# ```

# ## Fully Connected Layer, FC

# a status that every input node each connnected to every output node

# e.g. 2 dim data => 1 dim vector

# In[ ]:


import torch
from torch import nn


# In[ ]:


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc = nn.Linear(32*32*32, 10)
        
    def forward(self, x):
        x = self.conv1
        x = self.conv2
        x = torch.flatten(x)
        x = self.fc(x)
        
        return x


# In[ ]:


import torch
from torch import nn


# In[ ]:


class sentence_classifier(nn.Module):
    def __init__(self, pretrained_embedding, filter_sizes, max_length, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(pretrained_embedding, dtype=torch.float32)
        )
        embedding_dim = self.embedding.weight.shape[1]
        
        conv=[]
        
        for size in filter_sizes:
            conv.appendd(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels = embedding_dim,
                        out_channels=1,
                        kernel_size=size
                    ),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=max_length-size-1),
                )
            )
        self.conv_filters = nn.ModuleList(conv)
        
        output_size = len(filter_sizes)
        self.pre_classifier = nn.Linear(output_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_size, 1)
        
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings = embeddings.permute(0, 2, 1)
        
        conv_outputs = [conv(embeddings) for conv in self.conv_filters]
        concat_outputs = torch.cat([conv.squeeze(-1) for conv in conv_outputs], dim=1)
        
        logits = self.pre_classifier(concat_outputs)
        logits = self.dropout(logits)
        logits = self.classifier(logits)
        
        return logits


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
filter_sizes = [3, 3, 4, 4, 5, 5]
classifier = sentence_classifier(
    pretrained_embedding = init_embeddings,
    filter_sizes = filter_sizes,
    max_length = max_length
).to(device)


# In[ ]:


criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = opttim.Adam(classifier.parameters(0, lr=0.001))


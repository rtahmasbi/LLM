

# Basics

```py
num_classes = 3
targets = torch.randint(0, num_classes, (5,))
targets

>> tensor([2, 1, 0, 2, 0])

```


```py
import torch
num_labels = 3
targets = torch.randint(0, 2, (5, num_labels)).float()
targets

>> tensor([[0., 0., 1.],
        [1., 1., 0.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 0., 0.]])

```


```py
A = torch.randn(2, 7, 3, 4)
B = torch.randn(2, 7, 4, 6)

C = A @ B
# (2, 7, 3, 6)



torch.tril(torch.ones(4, 4))
# tensor([[1., 0., 0., 0.],
#         [1., 1., 0., 0.],
#         [1., 1., 1., 0.],
#         [1., 1., 1., 1.]])


torch.finfo(torch.float32).min  # -3.4028235e+38
torch.finfo(torch.float16).min  # -65504.0

####
emb_size: int = 768
max_length: int = 512
vocab_size: int = 500
word_to_embedding = nn.Embedding(max_length, emb_size)


########
x = torch.tensor([[1,2,3],[4,5,6]])
mask = x > 3
#mask = [[False False False], [ True  True  True]]

y = x.masked_fill(mask, 0)

print(y)
>> tensor([[1,2,3],[0,0,0]])



####
x = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)
n1 = nn.Linear(2,6)
y = n1(x)
y


####
logits = torch.randn(100, 128, 50257)
tokens = logits.argmax(dim=-1)[:20]

next_token = logits[:, -1, :].argmax(dim=-1)


#### squeeze
tokens = torch.tensor([[[2, 5, 7, 9]]])
tokens.squeeze() # tensor([2, 5, 7, 9])
# Returns a tensor with all specified dimensions of input of size 1 removed.
# (A×1×B×C×1×D) -> (A×B×C×D)

#### unsqueeze
#Returns a new tensor with a dimension of size one inserted at the specified position.
torch.tensor([1,2,3,4]).unsqueeze(0)
# tensor([[1, 2, 3, 4]])

torch.tensor([1,2,3,4]).unsqueeze(1)
# tensor([[1], [2], [3], [4]])

```


```py
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
# https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
torch.eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
torch.cat(tensors, dim=0, *, out=None) → Tensor
torch.stack(tensors, dim=0, *, out=None) → Tensor

torch.zeros(3,3)
torch.ones(3,3)
torch.eye(2,4)
torch.eye(4)
```


# Tensor Views
Supporting View avoids explicit data copy, thus allows us to do fast and memory efficient reshaping, slicing and element-wise operations.

```py
base = torch.tensor([[0, 1],[2, 3]])
base.is_contiguous() # True
t = base.transpose(0, 1)  # `t` is a view of `base`. No data movement happened here.
t.is_contiguous() # False
c = t.contiguous()

```

```py
t = torch.rand(4, 4)
b = t.view(2, 8)
t.storage().data_ptr() == b.storage().data_ptr()  # `t` and `b` share the same underlying data.
b[0][0] = 3.14
t[0][0] # tensor(3.14)
```

# permute vs transpose
```py
# For a 4D tensor x with shape (A, B, C, D), x.permute(0, 3, 1, 2) would result in a tensor with shape (A, D, B, C)
# For a 4D tensor x with shape (A, B, C, D), x.transpose(1, 2) would result in a tensor with shape (A, C, B, D). This is equivalent to x.permute(0, 2, 1, 3)
```

# simple linear regression
## linear model
```py
import torch
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


N = 100
x = torch.randn(N)
y = x * 3 + 2 + torch.randn(N) * 0.3

# parameters
a = torch.randn(1, requires_grad=True) # requires_grad because we need its gradient
b = torch.randn(1, requires_grad=True)

def model(x):
    return a * x + b

optimizer = optim.Adam([a,b], lr=0.01)
# or
#optimizer = optim.SGD([a,b], lr=0.01, momentum=0.9)


for epoch in range(1000):
    y_pred = model(x)
    loss = F.mse_loss(y_pred, y)
    #
    optimizer.zero_grad() # 1. Zero the parameter gradients
    loss.backward() # 2. compute gradients
    optimizer.step() # 3. update parameters
    if epoch % 100 == 0:
        print(f"{epoch}, {loss.item():.4f}, {a.item():.4f}, {b.item():.4f}")


print("a =", a.item())
print("b =", b.item())

```


## linear model with nn
```py

import torch
import random

import torch.nn as nn
import torch.optim as optim


N = 100
x = torch.randn(N)
y = x * 3 + 2 + torch.randn(N) * 0.3


criterion = nn.MSELoss()
criterion(x,y)

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        # self.c = nn.Parameter(torch.tensor(5.0), requires_grad=False)

    def forward(self, x):
        return self.a * x + self.b



model = LinearModel().to(device)

#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# Or using Adam
optimizer = optim.Adam(model.parameters(), lr=0.01)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(1000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    #
    optimizer.zero_grad() # 1. Zero the parameter gradients
    loss.backward() # 2. compute gradients
    optimizer.step() # 3. update parameters
    if epoch % 100 == 0:
        print(f"{epoch}, {loss.item():.4f}, {model.a.item():.4f}, {model.b.item():.4f}")


print("a =", model.a.item())
print("b =", model.b.item())

```


# DataLoader, TensorDataset
```py
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(x, y)
DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4, # parallel loading
    drop_last=True
)

# Move batches to GPU, not the dataset.
for epoch in range(10):
    model.train()
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        ...
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
```


# nn.Sequential
```py
# Define the model
model = nn.Sequential(
    nn.Linear(8, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
)
 
```

# model.zero_grad() vs optimizer.zero_grad()
sometimes you have multiple optimizers.

`model.zero_grad()` resets gradients for all parameters in the model.


# LogisticRegression
https://machinelearningmastery.com/building-a-logistic-regression-classifier-in-pytorch/

```py
class LogisticRegression(torch.nn.Module):    
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        #super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
    # make predictions
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

```


Binary classification
(or when multiple labels can be true)
Sigmoid
\sigma(x) = \frac{1}{1 + e^{-x}}
nn.BCEWithLogitsLoss()


Softmax
Multi-label classification
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
nn.CrossEntropyLoss()

softmax for 2 classes = sigmoid (a surprising but useful result).


# In PyTorch
```
CrossEntropyLoss = LogSoftmax + NLLLoss
BCEWithLogitsLoss = Sigmoid + BCE
```
## BCE (Binary Cross Entropy)
BCE is used for binary classification or multi-label classification.

The formula is:
```
L = -[y\log(p) + (1-y)\log(1-p)]
```
## NLLLoss (Negative Log Likelihood Loss)
NLLLoss is used for multi-class classification where exactly one class is correct.

The loss is:
```
L = -\log p(y)
```

## which loss to use?
For loss functions, `model_output.shape == loss_input.shape`


| Loss              | Problem type         | Output layer | Sum of probs = 1? | Input (model output) shape        | Target shape                | Output (after activation) shape |
| ----------------- | -------------------- | ------------ | ----------------- | --------------------------------- | --------------------------- | ------------------------------- |
| NLLLoss           | multi-class          | log-softmax  | Yes               | `(N, C)` or `(N, C, d1, ..., dk)` | `(N)` or `(N, d1, ..., dk)` | same as input `(N, C, ...)`     |
| CrossEntropyLoss  | multi-class          | raw logits   | Yes               | `(N, C)` or `(N, C, d1, ..., dk)` | `(N)` or `(N, d1, ..., dk)` | `(N, C, ...)` (after softmax)   |
| BCELoss           | binary / multi-label | sigmoid      | No                | `(N, *)`                          | `(N, *)`                    | `(N, *)` (after sigmoid)        |
| BCEWithLogitsLoss | binary / multi-label | raw logits   | No                | `(N, *)`                          | `(N, *)`                    | `(N, *)` (after sigmoid)        |



```py

torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)

torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
torch.nn.GaussianNLLLoss(*, full=False, eps=1e-06, reduction='mean')
torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False)
torch.nn.modules.loss.L1Loss(size_average=None, reduce=None, reduction='mean')
torch.nn.modules.loss.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
torch.nn.modules.loss.PoissonNLLLoss(log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')

torch.nn.modules.loss.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
# continuous (unsegmented) time series and a target sequence with C as the number of classes

```



# examples in detail
## regression (MSELoss)
```py
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple feedforward neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)  # Input size 10, hidden size 20
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)   # Output size 1

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model, loss function, and optimizer
model = SimpleNet()
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy input (batch of 5 samples, each with 10 features)
inputs = torch.randn(5, 10)

# Dummy target values (batch of 5 outputs)
targets = torch.randn(5, 1)

# Forward pass
outputs = model(inputs)

# Compute loss
loss = criterion(outputs, targets)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Loss:", loss.item())


```


## multi-label binary classification network (BCELoss + sigmoid)
```py
import torch
import torch.nn as nn
import torch.optim as optim

# Define a multi-label binary classification network
class MultiLabelNet(nn.Module):
    def __init__(self, input_size, num_labels):
        super(MultiLabelNet, self).__init__()
        self.fc = nn.Linear(input_size, num_labels)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # Sigmoid for multi-label probabilities

# Example configuration
input_size = 8   # Number of input features
num_labels = 4  # Number of independent binary labels

# Create model, loss, and optimizer
model = MultiLabelNet(input_size, num_labels)
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dummy input (batch of 3 samples, each with 8 features)
inputs = torch.randn(3, input_size)

# Dummy target (batch of 3 samples, each with 4 binary labels)
targets = torch.randint(0, 2, (3, num_labels)).float()

# Forward pass
outputs = model(inputs)

# Compute loss
loss = criterion(outputs, targets)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Outputs (probabilities):", outputs)
print("Loss:", loss.item())

```


## multi-label classifier (BCEWithLogitsLoss + NO sigmoid)
```py
import torch
import torch.nn as nn
import torch.optim as optim

# Multi-label classifier (NO sigmoid in the model)
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, num_labels):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_labels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)   # raw logits
        return x

# configuration
input_size = 10
num_labels = 4

model = MultiLabelClassifier(input_size, num_labels)

# BCE with logits
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

# dummy batch
inputs = torch.randn(5, input_size)

# multi-label targets (0 or 1 per label)
targets = torch.randint(0, 2, (5, num_labels)).float()

# forward
logits = model(inputs)

# loss
loss = criterion(logits, targets)

# backward
optimizer.zero_grad()
loss.backward()
optimizer.step()

# probabilities (apply sigmoid only for inference)
probs = torch.sigmoid(logits)

print("Logits:", logits)
print("Probabilities:", probs)
print("Loss:", loss.item())
```


## multi-class classification (CrossEntropyLoss + No softmax)
```py
import torch
import torch.nn as nn
import torch.optim as optim

# A simple neural network for multi-class classification
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)   # Hidden layer with 50 units
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, num_classes)  # Output layer with num_classes logits

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # No softmax here!
        return x

# Example configuration
input_size = 10   # Number of input features
num_classes = 4   # Number of classes to predict

# Model, loss, and optimizer
model = SimpleClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dummy input (batch of 5 samples, each with 10 features)
inputs = torch.randn(5, input_size)

# Dummy target (class indices from 0 to num_classes-1)
targets = torch.randint(0, num_classes, (5,))

# Forward pass (raw logits)
logits = model(inputs)

# Compute loss (CrossEntropyLoss applies softmax internally)
loss = criterion(logits, targets)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Logits:", logits)
print("Loss:", loss.item())

```


## multi-class classification (NLLLoss + LogSoftmax)
```py
import torch
import torch.nn as nn
import torch.optim as optim

# Neural network with log softmax in the model
class ClassifierWithLogSoftmax(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassifierWithLogSoftmax, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)  # Log softmax across classes

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.log_softmax(x)  # Outputs log probabilities
        return x

# Example configuration
input_size = 10
num_classes = 4

# Model, loss, and optimizer
model = ClassifierWithLogSoftmax(input_size, num_classes)
criterion = nn.NLLLoss()  # Negative Log Likelihood Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dummy input (batch of 5 samples, each with 10 features)
inputs = torch.randn(5, input_size)

# Dummy target (class indices from 0 to num_classes-1)
targets = torch.randint(0, num_classes, (5,))

# Forward pass (log probabilities)
log_probs = model(inputs)

# Compute loss (NLLLoss expects log probabilities)
loss = criterion(log_probs, targets)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Log probabilities:", log_probs)
print("Loss:", loss.item())

```



# AdamW vs Adam
## AdamW
```txt
Adam_update(gradient)
weight = weight - lr * weight_decay * weight
```

## Adam
```txt
gradient = gradient + weight_decay * weight
Adam_update(gradient)
```


# Autoencoder
https://github.com/maalvarezl/MLAI-Labs/blob/master/Lab%208%20-%20Unsupervised%20learning.ipynb

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

#Set the random seed for reproducibility 
torch.manual_seed(2020) 

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # 1 input image channel, 16 output channel, 3x3 square convolution
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  #to range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



mnist_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
# Training (optimisation) parameters
batch_size=64
learning_rate=1e-3
max_epochs = 20

# sChoose mean square error loss
criterion = nn.MSELoss() 
# Choose the Adam optimiser
optimizer = torch.optim.Adam(myAE.parameters(), lr=learning_rate, weight_decay=1e-5)
# Specify how the data will be loaded in batches (with random shuffling)
train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)
# Storage
outputs = []

# Start training
for epoch in range(max_epochs):
    for data in train_loader:
        img, label = data
        optimizer.zero_grad()
        recon = myAE(img)
        loss = criterion(recon, img)
        loss.backward()
        optimizer.step()            
    if (epoch % 2) == 0:
        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
    outputs.append((epoch, img, recon),)

```


https://docs.pytorch.org/docs/stable/nn.html
https://docs.pytorch.org/docs/stable/nn.functional.html
https://docs.pytorch.org/docs/stable/optim.html
https://docs.pytorch.org/docs/stable/distributed.html
gloo, mpi, nccl, xccl
https://docs.pytorch.org/docs/stable/cuda.html
https://docs.pytorch.org/docs/stable/data.html
- torch.utils.data
    - DataLoader
    - StackDataset
    - ConcatDataset
    - ChainDataset
    - Subset
    - random_split



# `nn.Sequential` vs `nn.ModuleList` vs `nn.ModuleDict`



# torchvision.models
```py
from torchvision import models
models.alexnet()
models.vgg11()
models.vgg13()
models.vgg16()
models.vgg19()

models.vgg11_bn()
models.vgg13_bn()
models.vgg16_bn()
models.vgg19_bn()

models.resnet18()
models.resnet34()
models.resnet50()
models.resnet101()
models.resnet152()

models.resnext50_32x4d()
models.resnext101_32x8d()
models.resnext101_64x4d()
```

# torchtext.models
```py
from torchtext.models import RobertaModel
from torchtext.models import ROBERTA_BASE_ENCODER
from torchtext.models import ROBERTA_LARGE_ENCODER

from torchtext.models import T5_BASE
from torchtext.models import T5_SMALL
from torchtext.models import T5_LARGE
from torchtext.models import T5_3B
from torchtext.models import T5_11B

from torchtext.models import TransformerEncoder
```

```py
from torchtext.models import TransformerEncoder

model = TransformerEncoder(
    vocab_size=30000,
    embed_dim=512,
    num_layers=6,
    num_heads=8
)
```


```py
from torchtext.models import TextClassificationModel

model = TextClassificationModel(
    vocab_size=10000,
    embed_dim=64,
    num_class=4
)
```

```py
import torchtext.models as models
print(dir(models))
```


```py
from transformers import AutoModel

from transformers.models.auto.configuration_auto import CONFIG_MAPPING

print(CONFIG_MAPPING.keys())


```

https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

https://github.com/karpathy/nanoGPT/blob/master/model.py






# ViterbiLoss
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling/
https://arxiv.org/pdf/2208.11867

https://www.hyperscience.ai/blog/exploring-conditional-random-fields-for-nlp-applications/

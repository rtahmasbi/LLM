
# TensorFlow
https://www.tensorflow.org/api_docs/python/tf/keras/Model


https://www.tensorflow.org/api_docs/python/tf/all_symbols

## Modules
### ft.math

### tf.nn
https://www.tensorflow.org/api_docs/python/tf/nn
```
avg_pool
tf.nn.conv1d
avg_pool1d
dropout
max_pool
max_pool1d
relu
softmax
```


### tf.keras

#### tf.keras.Model


```py
from tensorflow import keras
inputs = keras.Input(shape=(37,))
x = keras.layers.Dense(32, activation="relu")(inputs)
# output_shape=(*, 32), nparam: (37+1)*32
outputs = keras.layers.Dense(5, activation="softmax")(x)
# output_shape=(*, 5), nparam: (32+1)*5
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

```


# Reshape
```py
x = keras.Input(shape=(12,))
y = keras.layers.Reshape((3, 4))(x)
y.shape # (None, 3, 4)

y = keras.layers.Reshape((-1, 2, 2))(x)
y.shape # (None, 3, 2, 2)

```

# Flatten
```py
x = keras.Input(shape=(10, 64))
y = keras.layers.Flatten()(x)
y.shape # (None, 640)

x = keras.Input(shape=(10, 64, ))
y = keras.layers.Flatten()(x)
y.shape # (None, 640)

```


# This is very important
https://www.tensorflow.org/api_docs/python/tf/keras/Model
- With the "Functional API"
- By subclassing the Model class
- With the Sequential class


By subclassing the Model class: Once the model is created, you can config the model with losses and metrics with model.compile(), train the model with model.fit(), or use the model to do prediction with model.predict().


#### class Sequential
```py
model = keras.Sequential()
model.add(keras.Input(shape=(16,)))
model.add(keras.layers.Dense(8))
```

```
add
compile
compute_loss
evaluate
fit
get_layer
loss
predict
save
summary
test_step
train_step
```


## Classes

## Functions
```py
tf.math.abs(x, name=None)
```

https://www.tensorflow.org/resources/libraries-extensions



https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
It is GAN model.
change in the `train_step` and `with tf.GradientTape() as tape:`

################################################################################
################################################################################
################################################################################
# tf.keras.Input

all the same:
```python
Input(shape=(32,)).shape # TensorShape([None, 32])

Input(shape=(32)).shape  # TensorShape([None, 32])

Input(shape=32).shape    # TensorShape([None, 32])

```


################################################################################
################################################################################
################################################################################
# tf.data.Dataset
https://www.tensorflow.org/api_docs/python/tf/data/Dataset

```py
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = tf.data.TextLineDataset(["file1.txt", "file2.txt"])
dataset = tf.data.Dataset.list_files("/path/*.txt")
```

dataset = tf.data.Dataset.range(100)



```py
import tensorflow as tf

dataset = tf.data.Dataset.range(100)
def dataset_fn(ds):
  return ds.filter(lambda x: x < 5)
dataset = dataset.apply(dataset_fn)
list(dataset.as_numpy_iterator())

```

```
Dataset.apply
Dataset.as_numpy_iterator
Dataset.batch
Dataset.cache
Dataset.range
Dataset.cardinality
Dataset.concatenate
counter
enumerate
filter
fingerprint
flat_map
from_generator
from_tensor_slices
from_tensors
get_single_element
group_by_window
interleave
list_files
load
map
options
padded_batch
prefetch
ragged_batch
random
range
rebatch
reduce
rejection_resample
repeat
sample_from_datasets
save
scan
shard
shuffle
skip
snapshot
sparse_batch
take
take_while
unbatch
unique
window
Shift
Stride
with_options
zip
__len__
```

################################################################################
################################################################################
################################################################################
# simple model
```python
import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])


predictions = model(x_train[:1]).numpy()
predictions

d = np.random.rand(28,28)
d = np.expand_dims(d, axis=0)
predictions = model(d)
predictions


#The tf.nn.softmax function converts these logits to "probabilities" for each class:
tf.nn.softmax(predictions).numpy()


model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
#
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
probability_model(x_test[:5])


```


################################################################################
################################################################################
################################################################################
# Build the tf.keras model using the Keras model subclassing API:

```python
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


#
@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)



#

EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )  

```


## with class

```python
class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])

  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)

layer = MyDenseLayer(10)

layer(tf.zeros([10, 5])) # Calling the layer `.builds` it.

```





# TensorFlow Probability

https://www.tensorflow.org/probability

tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),



```python
import tensorflow as tf
import tensorflow_probability as tfp

# Pretend to load synthetic data set.
features = tfp.distributions.Normal(loc=0., scale=1.).sample(int(100e3))
labels = tfp.distributions.Bernoulli(logits=1.618 * features).sample()

# Specify model.
model = tfp.glm.Bernoulli()

# Fit model given data.
coeffs, linear_response, is_converged, num_iter = tfp.glm.fit(
    model_matrix=features[:, tf.newaxis],
    response=tf.cast(labels, dtype=tf.float32),
    model=model)
# ==> coeffs is approximately [1.618] (We're golden!)
```




## Structural Time Series - tfp.sts




# TensorFlow Decision Forests (TF-DF)
import tensorflow_decision_forests as tfdf

TensorFlow Decision Forests (TF-DF) is a collection of state-of-the-art algorithms for the training, serving and interpretation of Decision Forest models. The library is a collection of Keras models and supports classification, regression and ranking.




# tf.keras
## tf.keras.layers
```py
tf.keras.layers.AbstractRNNCell

tf.keras.layers.Cropping1D

tf.keras.layers.LSTM
tf.keras.layers.SimpleRNNCell

tf.keras.layers.Activation
tf.keras.layers.Cropping2D
tf.keras.layers.LSTMCell
tf.keras.layers.Softmax

tf.keras.layers.ActivityRegularization
tf.keras.layers.Cropping3D
tf.keras.layers.Lambda
tf.keras.layers.SpatialDropout1D

tf.keras.layers.Add
tf.keras.layers.Dense
tf.keras.layers.Layer
tf.keras.layers.SpatialDropout2D

tf.keras.layers.AdditiveAttention
tf.keras.layers.DenseFeatures
tf.keras.layers.LayerNormalization
tf.keras.layers.SpatialDropout3D

tf.keras.layers.AlphaDropout
tf.keras.layers.DepthwiseConv2D
tf.keras.layers.LeakyReLU
tf.keras.layers.StackedRNNCells

tf.keras.layers.Attention
tf.keras.layers.Dot
tf.keras.layers.LocallyConnected1D
tf.keras.layers.Subtract

tf.keras.layers.Average
tf.keras.layers.Dropout
tf.keras.layers.LocallyConnected2D
tf.keras.layers.ThresholdedReLU

tf.keras.layers.AveragePooling1D
tf.keras.layers.ELU
tf.keras.layers.Masking
tf.keras.layers.TimeDistributed

tf.keras.layers.AveragePooling2D


tf.keras.layers.Embedding


tf.keras.layers.MaxPool1D
tf.keras.layers.UpSampling1D

tf.keras.layers.AveragePooling3D
tf.keras.layers.Flatten
tf.keras.layers.MaxPool2D
tf.keras.layers.UpSampling2D

tf.keras.layers.AvgPool1D
tf.keras.layers.GRU
tf.keras.layers.MaxPool3D
tf.keras.layers.UpSampling3D

tf.keras.layers.AvgPool2D
tf.keras.layers.GRUCell
tf.keras.layers.MaxPooling1D
tf.keras.layers.Wrapper

tf.keras.layers.AvgPool3D
tf.keras.layers.GaussianDropout
tf.keras.layers.MaxPooling2D
tf.keras.layers.ZeroPadding1D

tf.keras.layers.BatchNormalization
tf.keras.layers.GaussianNoise
tf.keras.layers.MaxPooling3D
tf.keras.layers.ZeroPadding2D

tf.keras.layers.Bidirectional
tf.keras.layers.GlobalAveragePooling1D
tf.keras.layers.Maximum
tf.keras.layers.ZeroPadding3D

tf.keras.layers.Concatenate
tf.keras.layers.GlobalAveragePooling2D
tf.keras.layers.Minimum
tf.keras.layers.add

tf.keras.layers.Conv1D
tf.keras.layers.GlobalAveragePooling3D
tf.keras.layers.MultiHeadAttention
tf.keras.layers.average

tf.keras.layers.Conv1DTranspose
tf.keras.layers.GlobalAvgPool1D
tf.keras.layers.Multiply
tf.keras.layers.concatenate

tf.keras.layers.Conv2D
tf.keras.layers.GlobalAvgPool2D
tf.keras.layers.PReLU
tf.keras.layers.deserialize

tf.keras.layers.Conv2DTranspose
tf.keras.layers.GlobalAvgPool3D
tf.keras.layers.Permute
tf.keras.layers.dot

tf.keras.layers.Conv3D
tf.keras.layers.GlobalMaxPool1D
tf.keras.layers.RNN
tf.keras.layers.experimental
tf.keras.layers.Conv3DTranspose
tf.keras.layers.GlobalMaxPool2D
tf.keras.layers.ReLU
tf.keras.layers.maximum

tf.keras.layers.ConvLSTM2D
tf.keras.layers.GlobalMaxPool3D
tf.keras.layers.RepeatVector
tf.keras.layers.minimum

tf.keras.layers.Convolution1D
tf.keras.layers.GlobalMaxPooling1D
tf.keras.layers.Reshape
tf.keras.layers.multiply

tf.keras.layers.Convolution1DTranspose
tf.keras.layers.GlobalMaxPooling2D
tf.keras.layers.SeparableConv1D
tf.keras.layers.serialize

tf.keras.layers.Convolution2D
tf.keras.layers.GlobalMaxPooling3D
tf.keras.layers.SeparableConv2D
tf.keras.layers.subtract

tf.keras.layers.Convolution2DTranspose
tf.keras.layers.Input
tf.keras.layers.SeparableConvolution1D

tf.keras.layers.Convolution3D
tf.keras.layers.InputLayer
tf.keras.layers.SeparableConvolution2D

tf.keras.layers.Convolution3DTranspose
tf.keras.layers.InputSpec
tf.keras.layers.SimpleRNN
          
```




## tf.keras.losses
```py
tf.keras.losses.BinaryCrossentropy
tf.keras.losses.CategoricalCrossentropy 
tf.keras.losses.CategoricalHinge
tf.keras.losses.CosineSimilarity
tf.keras.losses.Hinge
tf.keras.losses.KLDivergence
tf.keras.losses.SparseCategoricalCrossentropy
```

## losses
https://keras.io/api/losses/
- binary_crossentropy
- CategoricalCrossentropy
- mean_absolute_error
- mean_squared_error



# SparseCategoricalCrossentropy vs CategoricalCrossentropy

categorical_crossentropy (cce) produces a one-hot array containing the probable match for each category,
sparse_categorical_crossentropy (scce) produces a category index of the most likely matching category.
Consider a classification problem with 5 categories (or classes).

In the case of cce, the one-hot target may be [0, 1, 0, 0, 0] and the model may predict [.2, .5, .1, .1, .1] (probably right)

In the case of scce, the target index may be [1] and the model may predict: [.5].


# tf.keras.layers.Activation
- relu       max(x, 0)
- sigmoid    sigmoid(x) = 1 / (1 + exp(-x))  for binary - Sigmoid is equivalent to a 2-element softmax,
- softmax    exp(x) / sum(exp(x)).
- softplus   softplus(x) = log(exp(x) + 1).
- softsign   softsign(x) = x / (abs(x) + 1).
- tanh       tanh(x) = sinh(x) / cosh(x)
- selu
- elu
- exponential
- leaky_relu
- relu6
- silu
- hard_silu
- gelu
- hard_sigmoid
- linear
- mish
- log_softmax
...


https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
- Binary Classification: One node, sigmoid activation.
- Multiclass Classification: One node per class, softmax activation.
- Multilabel Classification: One node per class, sigmoid activation.




# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

model.compile(loss='categorical_crossentropy', optimizer='adam')
or
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.01))



- Adadelta
- Adam
- RMSprop
- SGD
...



################################################################################
################################################################################
################################################################################
# tensorflow time series RNN
https://www.tensorflow.org/tutorials/structured_data/time_series



Very good
https://analyticsindiamag.com/comparing-arima-model-and-lstm-rnn-model-in-time-series-forecasting/



################################################################################
################################################################################
################################################################################
# embedding
https://keras.io/api/layers/core_layers/embedding/
https://stats.stackexchange.com/questions/270546/how-does-keras-embedding-layer-work

```py
# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
```


## good example for colors
https://cosmiccoding.com.au/tutorials/encoding_colours
```
data:
0	Grey	0.329412	0.329412	0.329412	0
1	Grey, Silver	0.752941	0.752941	0.752941	1
2	grey	0.745098	0.745098	0.745098	2
3	LightGray	0.827451	0.827451	0.827451	3
4	LightSlateGrey	0.466667	0.533333	0.600000	4
5	SlateGray	0.439216	0.501961	0.564706	5
6	SlateGray1	0.776471	0.886275	1.000000	6



model = keras.Sequential()
model.add(Embedding(num_colours, embedding_dims, input_length=2))
model.add(Lambda(sum_dist, output_shape=(1,), name="Dist"))
model.add(Dense(1, activation="sigmoid"))
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["mse"])

```

## another good example
https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/


https://www.kaggle.com/rajmehra03/a-detailed-explanation-of-keras-embedding-layer


The output embedding shape is (3,12,8).

3---> no of documents (sentences)

12---> each document is made of 12 words which was our maximum length of any document.

& 8---> each word is 8 dimensional.
A particular word in a specific document has embedding dim of 8


```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(20, 64, input_length=10))
#vocabulary size:20 the largest integer (i.e. word index) in the input should be no larger than 19 (vocabulary size). 
# model.output_shape is (None, 10, 64), where `None` is the batch dimension
input_array = np.random.randint(20, size=(32, 10))
model.compile(optimizer='rmsprop', loss='mse')
output_array = model.predict(input_array)
print(output_array.shape)
# (32, 10, 64)



```



sigmoid(x) = 1 / (1 + exp(-x)).


softmax = exp(x) / tf.reduce_sum(exp(x)).

tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0) = max(x, 0)


foo = tf.constant([-10, -5, 0.0, 5, 10], dtype = tf.float32)
tf.keras.activations.relu(foo).numpy()


# dense layer
units: Positive integer, dimensionality of the output space.
```
tf.keras.layers.Dense(
    units,
    activation=None,
    ...
)
```


```python
Multi-class single-label classification - MNIST
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
#
Regression to arbitrary values - Bosten Housing price prediction
The goal is to predict a single continuous value instead of a discrete label of the house price with given data.

# predict house price last Dense layer
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])



Regression to values between 0 and 1
# Jet engine health assessment last Dense layer
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])


```

Dense implements the operation: `output = activation(dot(input, kernel) + bias)` where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True). 


```python
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(16,)))
model.add(tf.keras.layers.Dense(32, activation='relu'))
# Now the model will take as input arrays of shape (None, 16)  
# and output arrays of shape (None, 32).  
# Note that after the first layer, you don't need to specify  
# the size of the input anymore:  
```



# Layers
https://keras.io/api/layers/core_layers/

## Input object

## Dense layer

## Activation layer
https://keras.io/api/layers/core_layers/activation/

```
>>> layer = tf.keras.layers.Activation('relu')
>>> output = layer([-3.0, -1.0, 0.0, 2.0])
>>> list(output.numpy())
[0.0, 0.0, 0.0, 2.0]
>>> layer = tf.keras.layers.Activation(tf.nn.relu)
>>> output = layer([-3.0, -1.0, 0.0, 2.0])
>>> list(output.numpy())
[0.0, 0.0, 0.0, 2.0]
```

## Embedding layer

## Masking layer


## Recurrent layers
https://keras.io/api/layers/recurrent_layers/

### LSTM layer
Long Short-Term Memory (LSTM) is an RNN architecture specifically designed to address the vanishing gradient problem. The key to the LSTM solution to the technical problems was the specific internal structure of the units used in the model.


### GRU layer
### SimpleRNN layer
### TimeDistributed layer
### Bidirectional layer
### ConvLSTM2D layer
### Base RNN layer



## Lambda layer

The Lambda layer exists so that arbitrary expressions can be used as a Layer when constructing Sequential and Functional API models
```python
# add a x -> x^2 layer
model.add(Lambda(lambda x: x ** 2))

# add a layer that returns the concatenation
# of the positive part of the input and
# the opposite of the negative part

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

model.add(Lambda(antirectifier))

```


```python

embedding_dims = 2
num_colours = 10
def sum_dist(x):
    n = tf.keras.backend.permute_dimensions(x, pattern=(1, 0, 2))
    a, b = n[0], n[1]
    return tf.keras.backend.sum((a - b)**2, axis=-1, keepdims=True)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(num_colours, embedding_dims, input_length=2))
model.add(tf.keras.layers.Lambda(sum_dist, output_shape=(1,), name="Dist"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
print(model.summary())


```


## Attention layers
https://keras.io/api/layers/attention_layers/
- MultiHeadAttention layer
- Attention layer
- AdditiveAttention layer



The convolutional layer is an important part of a CNN, and its main function is to extract features [14–17]. It uses convolution operators to convolute the input image and saves the convolution results to different channels of the convolution layer.



## Pooling layers
https://keras.io/api/layers/pooling_layers/
```
MaxPooling1D layer
MaxPooling2D layer
MaxPooling3D layer
AveragePooling1D layer
AveragePooling2D layer
AveragePooling3D layer
GlobalMaxPooling1D layer
GlobalMaxPooling2D layer
GlobalMaxPooling3D layer
GlobalAveragePooling1D layer
GlobalAveragePooling2D layer
GlobalAveragePooling3D layer
```

```py
input_shape = (2, 3, 4)
x = tf.random.normal(input_shape)
y = tf.keras.layers.GlobalAveragePooling1D()(x)
print(y.shape) #(2,4)

```
takes average on the second dimension



MaxPooling1D: sliding window max
GlobalMaxPooling1D: max over the middle dimension, one dimension less



## math
tf.math.log
tf.math.pow


## TFP Probabilistic Layers
import tensorflow as tf
import tensorflow_probability as tfp

tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),

tfp.layers.VariationalGaussianProcess

https://www.tensorflow.org/probability/examples/Linear_Mixed_Effects_Models


features = tfp.distributions.Normal(loc=0., scale=1.).sample(int(100e3))
labels = tfp.distributions.Bernoulli(logits=1.618 * features).sample()

model = tfp.glm.Bernoulli()
tfp.glm.fit


### MCMC
https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/mcmc


## Bayesian neural network to classify MNIST digits
https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/bayesian_neural_network.py



## other layers
```
tf.keras.layers.Softmax
tf.keras.layers.Reshape
tf.keras.layers.Reshape
keras.layers.Subtract
```



```py
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# Equivalent to subtracted = keras.layers.subtract([x1, x2])
subtracted = keras.layers.Subtract()([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)

```


# What do “compile”, “fit”, and “predict” do in Keras sequential models?

Great!!!
https://datascience.stackexchange.com/questions/46124/what-do-compile-fit-and-predict-do-in-keras-sequential-models



Let's first see what we need to do when we want to train a model.

First, we want to decide a model architecture, this is the number of hidden layers and activation functions, etc. (compile)
Secondly, we will want to train our model to get all the parameters to the correct value to map our inputs to our outputs. (fit)
Lastly, we will want to use this model to do some feed-forward passes to predict novel inputs. (predict)


check the example there


# fit vs evaluate
https://stackoverflow.com/questions/44843581/what-is-the-difference-between-model-fit-an-model-evaluate-in-keras


```python
# A simple regression model
model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.compile(loss='mse', optimizer='rmsprop')

# The fit() method - trains the model
model.fit(x, y, nb_epoch=1000, batch_size=100)

Epoch 1000/1000
200/200 [==============================] - 0s - loss: 0.0023

# The evaluate() method - gets the loss statistics
model.evaluate(x, y, batch_size=200)     
# returns: loss: 0.0022612824104726315

# The predict() method - predict the outputs for the given inputs
model.predict(np.expand_dims(x[:3],1)) 
# returns: [ 0.65680361],[ 0.70067143],[ 0.70482892]

```

```py
input_shape = (2, 3, 4)
#x = tf.random.normal(input_shape)
x = np.random.normal(size=input_shape)
x[:,:,None,:] # --> adds one more dimention by just adding them into a bracket
```





# multivariate forecast
https://www.analyticsvidhya.com/blog/2020/10/multivariate-multi-step-time-series-forecasting-using-stacked-lstm-sequence-to-sequence-autoencoder-in-tensorflow-2-0-keras/




# RNN

https://www.tensorflow.org/guide/keras/rnn



# one_hot
```python
from keras.preprocessing.text import one_hot

docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]

one_hot(docs[0], vocab_size)
# [21, 23]

```


# plot the model
plot_model(model, 'model.png', show_shapes=True)

utils.plot_model(autoencoder, show_shapes=True, expand_nested=True)

model.summary()



```python

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils

import matplotlib.pyplot as plt

NUM_ATOMS = 120  # Maximum number of atoms

ATOM_DIM = 11 # len(SMILE_CHARSET)  # Number of atom types
BOND_DIM = 4 + 1  # Number of bond types
LATENT_DIM = 435  # Size of the latent space

gconv_units=[9]
adjacency_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS)
feature_shape=(NUM_ATOMS, ATOM_DIM)
latent_dim=LATENT_DIM
dense_units=[512]
dropout_rate=0.0



class RelationalGraphConvLayer(keras.layers.Layer):
    def __init__(
        self,
        units=128,
        activation="relu",
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        #
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
    #
    def build(self, input_shape):
        bond_dim = input_shape[0][1]
        atom_dim = input_shape[1][2]
        
        self.kernel = self.add_weight(
            shape=(bond_dim, atom_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="W",
            dtype=tf.float32,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(bond_dim, 1, self.units),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                name="b",
                dtype=tf.float32,
            )
        self.built = True
    #
    def call(self, inputs, training=False):
        adjacency, features = inputs
        # Aggregate information from neighbors
        x = tf.matmul(adjacency, features[:, None, :, :])
        # Apply linear transformation
        x = tf.matmul(x, self.kernel)
        if self.use_bias:
            x += self.bias
        # Reduce bond types dim
        x_reduced = tf.reduce_sum(x, axis=1)
        # Apply non-linear transformation
        return self.activation(x_reduced)


##### encoder

adjacency = keras.layers.Input(shape=adjacency_shape)
features = keras.layers.Input(shape=feature_shape)

# Propagate through one or more graph convolutional layers
features_transformed = features
for units in gconv_units:
    features_transformed = RelationalGraphConvLayer(units)(
        [adjacency, features_transformed])

# Reduce 2-D representation of molecule to 1-D
x = keras.layers.GlobalAveragePooling1D()(features_transformed)

# Propagate through one or more densely connected layers
for units in dense_units:
    x = layers.Dense(units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

z_mean = layers.Dense(latent_dim, dtype="float32", name="z_mean")(x)
log_var = layers.Dense(latent_dim, dtype="float32", name="log_var")(x)

encoder = keras.Model([adjacency, features], [z_mean, log_var], name="encoder")

encoder.summary()

utils.plot_model(encoder, to_file='model_encoder.png', show_shapes=True, expand_nested=True)


##### decoder

dense_units=[128, 256, 512]
dropout_rate=0.2
latent_dim=LATENT_DIM
adjacency_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS)
feature_shape=(NUM_ATOMS, ATOM_DIM)



latent_inputs = keras.Input(shape=(latent_dim,))
x = latent_inputs
for units in dense_units:
    x = keras.layers.Dense(units, activation="tanh")(x)
    x = keras.layers.Dropout(dropout_rate)(x)

# Map outputs of previous layer (x) to [continuous] adjacency tensors (x_adjacency)
x_adjacency = keras.layers.Dense(tf.math.reduce_prod(adjacency_shape))(x)
x_adjacency = keras.layers.Reshape(adjacency_shape)(x_adjacency)
# Symmetrify tensors in the last two dimensions
x_adjacency = (x_adjacency + tf.transpose(x_adjacency, (0, 1, 3, 2))) / 2
x_adjacency = keras.layers.Softmax(axis=1)(x_adjacency)

# Map outputs of previous layer (x) to [continuous] feature tensors (x_features)
x_features = keras.layers.Dense(tf.math.reduce_prod(feature_shape))(x)
x_features = keras.layers.Reshape(feature_shape)(x_features)
x_features = keras.layers.Softmax(axis=2)(x_features)

decoder = keras.Model(
    latent_inputs, outputs=[x_adjacency, x_features], name="decoder"
)

utils.plot_model(decoder, to_file='model_decoder.png', show_shapes=True, expand_nested=True)


```




https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum

tf.math.reduce_sum(
    input_tensor, axis=None, keepdims=False, name=None
)


# Dense(1) vs Dense(2)
When there are 2 classes and you generally have P(c=1) + P(c=0) = 1 then

keras.layers.Dense(2, activation = 'softmax') 

keras.layers.Dense(1, activation = 'sigmoid')


softmax: exp(z_i)/ sum_i(exp(z_i)) i \in 1..K
sigmoid: exp(z)/(1+exp(z))


----
no activation = "linear" activation: a(x) = x





# Create models from scratch

## Modelling Strategies in TensorFlow 
https://towardsdatascience.com/model-sub-classing-and-custom-training-loop-from-scratch-in-tensorflow-2-cc1d4f10fb4e
I have the pdf


Sequential API
Functional API
Model Subclassing API

## Training Mechanism
- We open a for loop that will iterate over the number of epochs.
- For each epoch, we open another for loop that will iterate over the datasets, in batches (x, y).
- For each batch, we open GradientTape() scope.
- Inside this scope, we call the model, the forward pass, and compute the loss.
- Outside this scope, we retrieve the gradients of the weights of the model with regard to the loss.
- Next, we use the optimizer to update the weights of the model based on the gradients.

https://keras.io/guides/writing_a_training_loop_from_scratch/


--important
https://keras.io/guides/customizing_what_happens_in_fit/



## new layer
if you want to define a new layer

```
class RelationalGraphConvLayer(keras.layers.Layer):
def __init__
def build(self, input_shape)
def call(self, inputs, training=False)
```

## new keras.Model
```py
class MoleculeGenerator(keras.Model):
def __init__:
def train_step(self, data):
def inference:
def call:
def build_graph:

or
@tf.function
def train_step(x, y):
@tf.function
def test_step(x, y):  

# example

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)


class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):
        self.add_loss(1e-2 * tf.reduce_sum(inputs))
        return inputs

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
        # Add any extra losses created during the forward pass.
        loss_value += sum(model.losses)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value
    

```


# SMILES
https://www.epa.gov/sites/default/files/2015-05/documents/appendf.pdf


Molecules can naturally be expressed as undirected graphs G = (V, E), where V is a set of vertices (atoms), and E a set of edges (bonds). As for this implementation, each graph (molecule) will be represented as an adjacency tensor A, which encodes existence/non-existence of atom-pairs with their one-hot encoded bond types stretching an extra dimension, and a feature tensor H, which for each atom, one-hot encodes its atom type. Notice, as hydrogen atoms can be inferred by RDKit, hydrogen atoms are excluded from A and H for easier modeling.

SMILE_CHARSET = '["C", "B", "F", "I", "H", "O", "N", "S", "P", "Cl", "Br"]'
ATOM_DIM = len(SMILE_CHARSET)  # Number of atom types
NUM_ATOMS = 120  # Maximum number of atoms


features: 2D one hot with 120*11 (11=ATOM_DIM)


# good example

```py
import numpy as np
import tensorflow as tf


model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(2, )),
        tf.keras.layers.Dense(2),
    ]
)
model.compile(loss='mse')

W = model.trainable_variables[0]
W.assign(np.array([[1.0, 0.0], [0.0, 1.0]]).T)

input = np.array([[1.0, 2.0], [3.0, 4.0], ], dtype=np.float32)

print("__call__:")
print(model(input))

print("Call:")
print(model.call(tf.convert_to_tensor(input)))

print("Predict:")
print(model.predict(input))
```



# attention
https://dmol.pub/dl/attention.html

https://keras.io/examples/vision/attention_mil_classification/
https://towardsdatascience.com/visual-attention-model-in-deep-learning-708813c2912c
I have the pdf

https://ion-mosnoi.medium.com/all-you-need-is-attention-computer-vision-edition-dbe7538330a4
I have the pdf

https://github.com/johnsmithm/multi-heads-attention-image-classification/blob/master/multi-heads-attention-mnist.py

https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a


https://blog.paperspace.com/image-classification-with-attention/

https://machinelearningmastery.com/the-attention-mechanism-from-scratch/


https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392



```python

from keras.layers import Dense, Layer, Dropout,  Conv2D, Input, Lambda, Flatten, TimeDistributed
from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
from keras.models import Model
from keras import backend as K

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.utils.layer_utils import get_source_inputs
import tensorflow as tf
from keras.callbacks import TensorBoard

def MultiHeadsAttModel(l=8*8, d=512, dv=64, dout=512, nv=8):
    v1 = Input(shape = (l, d))
    q1 = Input(shape = (l, d))
    k1 = Input(shape = (l, d))
    #
    v2 = Dense(dv*nv, activation = "relu")(v1)
    q2 = Dense(dv*nv, activation = "relu")(q1)
    k2 = Dense(dv*nv, activation = "relu")(k1)
    #
    v = Reshape([l, nv, dv])(v2)
    q = Reshape([l, nv, dv])(q2)
    k = Reshape([l, nv, dv])(k2)
    #
    att = Lambda(lambda x: K.batch_dot(x[0],x[1] ,axes=[-1,-1]) / np.sqrt(dv),
                 output_shape=(l, nv, nv))([q,k])# l, nv, nv
    att = Lambda(lambda x:  K.softmax(x) , output_shape=(l, nv, nv))(att)
    #
    out = Lambda(lambda x: K.batch_dot(x[0], x[1],axes=[4,3]),  output_shape=(l, nv, dv))([att, v])
    out = Reshape([l, d])(out)
    #
    out = Add()([out, q1])
    out = Dense(dout, activation = "relu")(out)
    return  Model(inputs=[q1,k1,v1], outputs=out)


class NormL(Layer):
    def __init__(self, **kwargs):
        super(NormL, self).__init__(**kwargs)
    #
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.a = self.add_weight(name='kernel', 
                                      shape=(1,input_shape[-1]),
                                      initializer='ones',
                                      trainable=True)
        self.b = self.add_weight(name='kernel', 
                                      shape=(1,input_shape[-1]),
                                      initializer='zeros',
                                      trainable=True)
        super(NormL, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x):
        eps = 0.000001
        mu = K.mean(x, keepdims=True, axis=-1)
        sigma = K.std(x, keepdims=True, axis=-1)
        ln_out = (x - mu) / (sigma + eps)
        return ln_out*self.a + self.b
    
    def compute_output_shape(self, input_shape):
        return input_shape


#

nb_classes = 10

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)

X_train = X_train.reshape(60000, 28,28,1)
X_test = X_test.reshape(10000, 28,28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

inp = Input(shape = (28,28,1)) # TensorShape([None, 28, 28, 1])
x = Conv2D(32,(2,2),activation='relu', padding='same')(inp)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64,(2,2),activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
x = Conv2D(64*3,(2,2),activation='relu')(x)

if True:
    x = Reshape([6*6,64*3])(x)    
    att = MultiHeadsAttModel(l=6*6, d=64*3 , dv=8*3, dout=32, nv = 8 )
    x = att([x,x,x])
    x = Reshape([6,6,32])(x)   
    x = NormL()(x)

x = Flatten()(x) 
x = Dense(256, activation='relu')(x)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=inp, outputs=x)
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

tbCallBack = TensorBoard(log_dir='./Graph/mhatt1', histogram_freq=0, write_graph=True, write_images=True)

model.fit(X_train, Y_train,
          batch_size=128, 
          epochs=100,
          verbose=1,          
          validation_data=(X_test, Y_test),
          callbacks=[tbCallBack]
         )


#######################
# test

inp = Input(shape = (28,28,1)) # TensorShape([None, 28, 28, 1])
x = Conv2D(32,(2,2),activation='relu', padding='same')(inp)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64,(2,2),activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
x = Conv2D(64*3,(2,2),activation='relu')(x) # TensorShape([None, 6, 6, 192])

x = Reshape([6*6,64*3])(x)
x.shape # TensorShape([None, 36, 192])
att = MultiHeadsAttModel(l=6*6, d=64*3 , dv=8*3, dout=32, nv = 8 )
x = att([x,x,x])

l=6*6
d=64*3
dv=8*3
dout=32
nv = 8

v1 = Input(shape = (l, d)) #TensorShape([None, 36, 192])
q1 = Input(shape = (l, d))
k1 = Input(shape = (l, d))
#
v2 = Dense(dv*nv, activation = "relu")(v1)
q2 = Dense(dv*nv, activation = "relu")(q1)
k2 = Dense(dv*nv, activation = "relu")(k1) # TensorShape([None, 36, 192])
#
v = Reshape([l, nv, dv])(v2) # TensorShape([None, 36, 8, 24])
q = Reshape([l, nv, dv])(q2)
k = Reshape([l, nv, dv])(k2)

att = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[-1,-1]) / np.sqrt(dv), output_shape=(l, nv, nv))([q2,k2])# l, nv, nv  but TensorShape([None, 36, 8, 36, 8])
att = Lambda(lambda x: K.softmax(x), output_shape=(l, nv, nv))(att) # TensorShape([None, 36, 8, 36, 8])
out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[4,3]), output_shape=(l, nv, dv))([att, v])


att = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[-1,-1]) / np.sqrt(dv))([q2,k2])
# (None, 36, 36)
att = Lambda(lambda x: K.softmax(x))(att) # (None, 36, 36)

out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[4,3]), output_shape=(l, nv, dv))([att, v2])


```

x_batch = K.ones(shape=(l, nv, dv))
y_batch = K.ones(shape=(l, nv, dv))
xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[-1, -1])
xy_batch_dot.shape


Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[-1,-1]) / np.sqrt(dv), output_shape=(l, nv, nv))([q,k])

K.batch_dot(q, k, axes=[-1,-1]).shape



## another example with images
https://keras.io/examples/vision/patch_convnet/



## good example
https://www.tensorflow.org/text/tutorials/transformer


## good simple video
https://www.youtube.com/watch?v=oaV_Fv5DwUM&ab_channel=ShuyiWang
bidirectional LSTM with attention to classify sentences by sentiment.


## keras attention code
https://keras.io/api/layers/attention_layers/attention/

https://github.com/keras-team/keras/blob/v3.3.3/keras/src/layers/attention/attention.py#L7

https://keras.io/api/layers/attention_layers/multi_head_attention/


# other tenserflow
https://www.tensorflow.org/resources/libraries-extensions

import tensorflow_federated as tff

https://www.tensorflow.org/probability
import tensorflow_probability as tfp




# advanced
```py

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")


train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)



# Keras model subclassing API:
class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)
    #
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


# Create an instance of the model
model = MyModel()


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# Use tf.GradientTape to train the model:

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    train_loss(loss)
    train_accuracy(labels, predictions)


# Test the model:
@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
  

#
EPOCHS = 2

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()
    #
    for images, labels in train_ds:
        train_step(images, labels)
    #
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    #
    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result():0.2f}, '
        f'Accuracy: {train_accuracy.result() * 100:0.2f}, '
        f'Test Loss: {test_loss.result():0.2f}, '
        f'Test Accuracy: {test_accuracy.result() * 100:0.2f}'
    )


```


GradientTape is a mathematical tool for automatic differentiation (autodiff), which is the core functionality of TensorFlow. 


eager execution

sess.run


stateful=True is usually used when you want to treat consecutive batches as consequtive inputs. In this case model is treating consequtive batches the same as it were in the same batch. 



https://medium.com/@bjorn_sing/tensorflow-gradient-tape-mnist-536c47fb8d85

```py

num_epochs = 30
batchsize = 64

model = create_my_model()
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
for epoch in range(num_epochs):
  for step in range(X_train.shape[0]//batchsize):
    start_idx = batchsize*step
    end_idx = batchsize*(step+1)
    X_batch = X_train[start_idx:end_idx]
    y_batch = y_train[start_idx:end_idx]
    with tf.GradientTape() as tape:
      pred = model(X_batch)
      loss = tf.keras.losses.categorical_crossentropy(y_batch, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    
    
```



# great


https://www.kaggle.com/code/yashbajpai/websiteclassification-with-ml-and-dl

SparseCategoricalCrossentropy
SparseTopKCategoricalAccuracy

```py
import tensorflow_hub as hub
embed = hub.KerasLayer("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2",
                 trainable = False,
                 name = 'universal_sentence_encoder')
```


# categorical_crossentropy vs sparse_categorical_crossentropy
categorical_crossentropy (cce) produces a one-hot array containing the probable match for each category,
sparse_categorical_crossentropy (scce) produces a category index of the most likely matching category.

n the case of cce, the one-hot target may be [0, 1, 0, 0, 0] and the model may predict [.2, .5, .1, .1, .1] (probably right)

In the case of scce, the target index may be [1] and the model may predict: [.5].


Many categorical models produce scce output because you save space, but lose A LOT of information (for example, in the 2nd example, index 2 was also very close.) I generally prefer cce output for model reliability.




# code embedding
https://huggingface.co/Salesforce/codet5p-110m-embedding
Supported languages (9 in total) are as follows: c, c++, c-sharp, go, java, javascript, php, python, ruby.

```py
checkpoint = "Salesforce/codet5p-110m-embedding"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

inputs = tokenizer.encode("def print_hello_world():\tprint('Hello World!')", return_tensors="pt").to(device)
embedding = model(inputs)[0]
````




# check this one for MultiLabelBinarizer
https://colab.research.google.com/drive/1d1WwB7pWgTMjkBNeYzpD8kUvuaYJaCvR#scrollTo=0ea_lUdYiygW

```py
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit_transform([{'sci-fi', 'thriller'}, {'comedy'}])
# array([[0, 1, 1], [1, 0, 0]])
list(mlb.classes_)
# ['comedy', 'sci-fi', 'thriller']
```



# multiple outputs with different optimizers
https://www.tensorflow.org/guide/keras/training_with_built_in_methods
mnist



# tf.keras.applications
https://www.tensorflow.org/api_docs/python/tf/keras/applications
https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50



# history
metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()




# time series


https://www.tensorflow.org/tutorials/structured_data/time_series


https://otexts.com/fpp2/accuracy.html
MAE
RMSE
MAPE
sMAPE
MASE - mean absolute scaled error



https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
https://keras.io/examples/timeseries/timeseries_classification_transformer/
https://keras.io/examples/timeseries/event_classification_for_payment_card_fraud_detection/


```py
import temporian as tp  # To convert transactions into tabular data
goup by CUSTOMER_ID, TERMINAL_ID
per terminal and 
```


https://keras.io/examples/timeseries/timeseries_anomaly_detection/
Conv1D,
predict model, 
use a thershold


https://keras.io/examples/timeseries/timeseries_traffic_forecasting/
GraphConv
LSTM
Dense


https://keras.io/api/ops/

https://keras.io/examples/timeseries/timeseries_weather_forecasting/
The model is shown data for first 5 days i.e. 720 observations, that are sampled every hour. The temperature after 72 (12 hours * 6 observation per hour) observation will be used as a label.


## keras.preprocessing.timeseries_dataset_from_array
Creates a dataset of sliding windows over a timeseries provided as array.



# graphs

## Graph Neural Networks
https://distill.pub/2021/gnn-intro/

## Graph Neural Networks
https://keras.io/examples/graph/gnn_citations/
https://arxiv.org/pdf/2011.08843v2

print(papers.subject.value_counts())
>
Neural_Networks           818
Probabilistic_Methods     426
Genetic_Algorithms        418
Theory                    351
Case_Based                298
Reinforcement_Learning    217
Rule_Learning             180
Name: subject, dtype: int64


## Graph attention network
https://keras.io/examples/graph/gat_node_classification/




# multiple loss

https://pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/

```py
# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
	"category_output": "categorical_crossentropy",
	"color_output": "categorical_crossentropy",
}
lossWeights = {"category_output": 1.0, "color_output": 1.0}
# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
	metrics=["accuracy"])

```




# tensorflow_decision_forests
https://www.tensorflow.org/decision_forests/tutorials/beginner_colab

import tensorflow_decision_forests as tfdf


# data
Sumbolic math
https://github.com/facebookresearch/SymbolicMathematics
It was used for graph NN



# RNN
https://www.tensorflow.org/guide/keras/working_with_rnns

## decoder encoder
output, state_h, state_c = layers.LSTM(64, return_state=True, name="encoder")(
    encoder_embedded
)
encoder_state = [state_h, state_c]

model = keras.Model([encoder_input, decoder_input], output)



## Bidirectional RNNs
model.add(
    layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(5, 10))
)



# Writing a training loop from scratch
https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
GradientTape


# Model Garden overview
https://www.tensorflow.org/tfmodels/nlp



# RLHF
https://huggingface.co/blog/stackllama
https://github.com/l294265421/alpaca-rlhf

https://github.com/michaelnny/InstructLLaMA



https://sudhirpol522.medium.com/reward-model-training-6d1693e41962

https://notesonai.com/rlhf+-+reinforcement+learning+with+human+feedback

https://notesonai.com/ppo+-+proximal+policy+optimization
https://notesonai.com/trpo+-+trust-region+policy+optimization
https://notesonai.com/benchmarking+model-based+reinforcement+learning



# Judges
https://arxiv.org/pdf/2311.09476




# Migrate multi-worker CPU/GPU training 
https://www.tensorflow.org/guide/migrate/multi_worker_cpu_gpu_training

```py
# Find ports that are available for the `'chief'` (the coordinator),
# `'worker'`s, and `'ps'` (parameter servers).
import portpicker

chief_port = portpicker.pick_unused_port()
worker_ports = [portpicker.pick_unused_port() for _ in range(3)]
ps_ports = [portpicker.pick_unused_port() for _ in range(2)]

# Dump the cluster information to `'TF_CONFIG'`.
tf_config = {
    'cluster': {
        'chief': ["localhost:%s" % chief_port],
        'worker': ["localhost:%s" % port for port in worker_ports],
        'ps':  ["localhost:%s" % port for port in ps_ports],
    },
    'task': {'type': 'chief', 'index': 0}
}
os.environ['TF_CONFIG'] = json.dumps(tf_config)

# Use a cluster resolver to bridge the information to the strategy created below.
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
```




# https://keras.io/api/applications/
- ResNet50
- VGG16


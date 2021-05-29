import os

import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt

data_dir = './sentiment labelled sentences'


def load_data(file_path):
    texts = []
    labels = []
    with open(file_path) as f:
        for line in f.readlines():
            text, label = line.strip().split('\t')
            texts.append(text)
            labels.append(label)
    df = pd.DataFrame({'text': texts, 'label': labels})
    return df


def explore_data(df):
    print(df.head())
    print(df.groupby('label').count())
    text_len = df['text'].map(lambda x: len(x.split()))
    text_len.hist(bins=20)


# Loading data and simple analysis
print('load and explore amazon data:')
amazon_df = load_data(os.path.join(data_dir, 'amazon_cells_labelled.txt'))
explore_data(amazon_df)
plt.show()
print('load and explore imdb data:')
imdb_df = load_data(os.path.join(data_dir, 'imdb_labelled.txt'))
explore_data(imdb_df)
plt.show()
print('load and explore yelp data:')
yelp_df = load_data(os.path.join(data_dir, 'yelp_labelled.txt'))
explore_data(yelp_df)
plt.show()

# Union all the sentences, split into train and test dataset
all_df = pd.concat([amazon_df, imdb_df, yelp_df]).drop_duplicates()
test_df = all_df.sample(frac=0.2, random_state=6)
train_df = all_df.append(test_df).drop_duplicates(keep=False)
print(test_df.shape)
print(train_df.shape)

# Convert text to word vector
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

maxlen = 30
max_words = 3000

train_x = train_df['text']
train_y = train_df['label']

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_x)

train_x = tokenizer.texts_to_sequences(train_x)
train_x = pad_sequences(train_x, maxlen=maxlen)
train_y = train_y.astype(float).values

test_x = test_df['text']
test_y = test_df['label']

test_x = tokenizer.texts_to_sequences(test_x)
test_x = pad_sequences(test_x, maxlen=maxlen)
test_y = test_y.astype(float).values

# Base model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.optimizers import Adam

epochs = 20
batch_size = 32

model = Sequential()
model.add(Embedding(max_words, 64, input_length=maxlen, name='embedding'))
model.add(Flatten(name='feature'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Use t-sne to visualize extracted feature
from sklearn import manifold

feature_extract_model = Model(inputs=model.input, outputs=model.get_layer('feature').output)
feature_extract = feature_extract_model.predict(train_x)
tsne = manifold.TSNE(random_state=6, init='pca')
x_tsne = tsne.fit_transform(feature_extract)
x_min, x_max = x_tsne.min(0), x_tsne.max(0)
x_norm = (x_tsne - x_min) / (x_max - x_min)
plt.figure(figsize=(6, 6))
for i in range(x_norm.shape[0]):
    plt.text(x_norm[i, 0], x_norm[i, 1], str(train_y[i]), color=plt.cm.Set1(train_y[i]),
             fontdict={'weight': 'bold', 'size': 9})

# Loss and accuracy curve
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

x_ticks = range(epochs)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(x_ticks, loss, 'bo', label='Training loss')
plt.plot(x_ticks, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')

plt.subplot(122)
plt.plot(x_ticks, acc, 'bo', label='Training acc')
plt.plot(x_ticks, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.legend()

# Evaluate model use test data
from tensorflow.math import confusion_matrix

test_predict = np.array(list(map(lambda x: 1 if x > 0.5 else 0, model.predict(test_x))))
cm = confusion_matrix(test_y, test_predict)
acc = model.evaluate(test_x, test_y)
print('Test loss is: {} and accuracy is: {}'.format(acc[0], acc[1]))
print('Precesion of label 0 is: {}, Recall of label 0 is: {}'.format(cm[0, 0] / (cm[0, 0] + cm[1, 0]),
                                                                     cm[0, 0] / (cm[0, 0] + cm[0, 1])))
print('Precesion of label 1 is: {}, Recall of label 1 is: {}'.format(cm[1, 1] / (cm[1, 1] + cm[0, 1]),
                                                                     cm[1, 1] / (cm[1, 1] + cm[1, 0])))
plt.figure(figsize=(12, 3))
plt.subplot(131)
plt.title('Predict Confusion Matrix')
sn.heatmap(cm, annot=True, fmt='.10g', cmap=plt.cm.Blues)
plt.xlabel('Predict label')
plt.ylabel('True label')

plt.subplot(132)
precision_cm = np.copy(cm)
precision_cm = precision_cm / precision_cm.sum(axis=0)
plt.title('Precision Confusion Matrix')
sn.heatmap(precision_cm, annot=True, fmt='.10g', cmap=plt.cm.Blues)
plt.xlabel('Predict label')
plt.ylabel('True label')

plt.subplot(133)
recall_cm = np.copy(cm)
recall_cm = recall_cm / recall_cm.sum(axis=1).reshape((2, 1))
plt.title('Recall Confusion Matrix')
sn.heatmap(recall_cm, annot=True, fmt='.10g', cmap=plt.cm.Blues)
plt.xlabel('Predict label')
plt.ylabel('True label')
plt.show()

# ----------------------------------------------------------------------------------------------
# combine all the step
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.math import confusion_matrix
from sklearn import manifold

# Loading data and simple analysis
print('load and explore amazon data:')
amazon_df = load_data(os.path.join(data_dir, 'amazon_cells_labelled.txt'))
explore_data(amazon_df)
plt.show()
print('load and explore imdb data:')
imdb_df = load_data(os.path.join(data_dir, 'imdb_labelled.txt'))
explore_data(imdb_df)
plt.show()
print('load and explore yelp data:')
yelp_df = load_data(os.path.join(data_dir, 'yelp_labelled.txt'))
explore_data(yelp_df)
plt.show()

# Union all the sentences, split into train and test dataset
all_df = pd.concat([amazon_df, imdb_df, yelp_df]).drop_duplicates()
test_df = all_df.sample(frac=0.2, random_state=6)
train_df = all_df.append(test_df).drop_duplicates(keep=False)
print(test_df.shape)
print(train_df.shape)

# Convert text to word vector
maxlen = 30
max_words = 3000

train_x = train_df['text']
train_y = train_df['label']

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_x)

train_x = tokenizer.texts_to_sequences(train_x)
train_x = pad_sequences(train_x, maxlen=maxlen)
train_y = train_y.astype(float).values

test_x = test_df['text']
test_y = test_df['label']

test_x = tokenizer.texts_to_sequences(test_x)
test_x = pad_sequences(test_x, maxlen=maxlen)
test_y = test_y.astype(float).values

# Build model and train
epochs = 20
batch_size = 32

model = Sequential()
model.add(Embedding(max_words, 64, input_length=maxlen, name='embedding'))
model.add(Flatten(name='feature'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Use t-sne to visualize extracted feature
feature_extract_model = Model(inputs=model.input, outputs=model.get_layer('feature').output)
feature_extract = feature_extract_model.predict(train_x)
tsne = manifold.TSNE(random_state=6, init='pca')
x_tsne = tsne.fit_transform(feature_extract)
x_min, x_max = x_tsne.min(0), x_tsne.max(0)
x_norm = (x_tsne - x_min) / (x_max - x_min)
plt.figure(figsize=(6, 6))
for i in range(x_norm.shape[0]):
    plt.text(x_norm[i, 0], x_norm[i, 1], str(train_y[i]), color=plt.cm.Set1(train_y[i]),
             fontdict={'weight': 'bold', 'size': 9})

# Loss and accuracy curve
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

x_ticks = range(epochs)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(x_ticks, loss, 'bo', label='Training loss')
plt.plot(x_ticks, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')

plt.subplot(122)
plt.plot(x_ticks, acc, 'bo', label='Training acc')
plt.plot(x_ticks, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.legend()

# Evaluate model use test data
test_predict = np.array(list(map(lambda x: 1 if x > 0.5 else 0, model.predict(test_x))))
cm = confusion_matrix(test_y, test_predict)
acc = model.evaluate(test_x, test_y)
print('Test loss is: {} and accuracy is: {}'.format(acc[0], acc[1]))
print('Precesion of label 0 is: {}, Recall of label 0 is: {}'.format(cm[0, 0] / (cm[0, 0] + cm[1, 0]),
                                                                     cm[0, 0] / (cm[0, 0] + cm[0, 1])))
print('Precesion of label 1 is: {}, Recall of label 1 is: {}'.format(cm[1, 1] / (cm[1, 1] + cm[0, 1]),
                                                                     cm[1, 1] / (cm[1, 1] + cm[1, 0])))
plt.figure(figsize=(12, 3))
plt.subplot(131)
plt.title('Predict Confusion Matrix')
sn.heatmap(cm, annot=True, fmt='.10g', cmap=plt.cm.Blues)
plt.xlabel('Predict label')
plt.ylabel('True label')

plt.subplot(132)
precision_cm = np.copy(cm)
precision_cm = precision_cm / precision_cm.sum(axis=0)
plt.title('Precision Confusion Matrix')
sn.heatmap(precision_cm, annot=True, fmt='.10g', cmap=plt.cm.Blues)
plt.xlabel('Predict label')
plt.ylabel('True label')

plt.subplot(133)
recall_cm = np.copy(cm)
recall_cm = recall_cm / recall_cm.sum(axis=1).reshape((2, 1))
plt.title('Recall Confusion Matrix')
sn.heatmap(recall_cm, annot=True, fmt='.10g', cmap=plt.cm.Blues)
plt.xlabel('Predict label')
plt.ylabel('True label')
plt.show()

# ---------------------------------------------------------------------------
# Abstract each function into a method to facilitate model tuning
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Flatten, Dense

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

from tensorflow.math import confusion_matrix
from sklearn import manifold


# process raw data into train_x, train_y, test_x, test_y
def preprocess_data(all_df, test_split=0.2, maxlen=20, max_words=10000):
    test_df = all_df.sample(frac=test_split, random_state=6)
    train_df = all_df.append(test_df).drop_duplicates(keep=False)
    print(test_df.shape)
    print(train_df.shape)

    train_x = train_df['text']
    train_y = train_df['label']

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_x)

    train_x = tokenizer.texts_to_sequences(train_x)
    train_x = pad_sequences(train_x, maxlen=maxlen)
    train_y = train_y.astype(float).values

    test_x = test_df['text']
    test_y = test_df['label']

    test_x = tokenizer.texts_to_sequences(test_x)
    test_x = pad_sequences(test_x, maxlen=maxlen)
    test_y = test_y.astype(float).values
    return train_x, train_y, test_x, test_y


# building the model
def build_model1(max_words, maxlen):
    model = Sequential()
    model.add(Embedding(max_words, 64, input_length=maxlen))
    model.add(Flatten(name='feature'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model


# Train the model with training data and validate the model with test data
def train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=20, batch_size=32, validation_split=0.2):
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    feature_extract_model = Model(inputs=model.input, outputs=model.get_layer('feature').output)
    feature_extract = feature_extract_model.predict(train_x)
    tsne = manifold.TSNE(random_state=6, init='pca')
    x_tsne = tsne.fit_transform(feature_extract)
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    x_norm = (x_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(6, 6))
    for i in range(x_norm.shape[0]):
        plt.text(x_norm[i, 0], x_norm[i, 1], str(train_y[i]), color=plt.cm.Set1(train_y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.show()

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    x_ticks = range(epochs)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(x_ticks, loss, 'bo', label='Training loss')
    plt.plot(x_ticks, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')

    plt.subplot(122)
    plt.plot(x_ticks, acc, 'bo', label='Training acc')
    plt.plot(x_ticks, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation acc')
    plt.legend()
    plt.show()

    test_predict = np.array(list(map(lambda x: 1 if x > 0.5 else 0, model.predict(test_x))))
    cm = confusion_matrix(test_y, test_predict)
    acc = model.evaluate(test_x, test_y)
    print('Test loss is: {} and accuracy is: {}'.format(acc[0], acc[1]))
    print('Precesion of label 0 is: {}, Recall of label 0 is: {}'.format(cm[0, 0] / (cm[0, 0] + cm[1, 0]),
                                                                         cm[0, 0] / (cm[0, 0] + cm[0, 1])))
    print('Precesion of label 1 is: {}, Recall of label 1 is: {}'.format(cm[1, 1] / (cm[1, 1] + cm[0, 1]),
                                                                         cm[1, 1] / (cm[1, 1] + cm[1, 0])))
    plt.figure(figsize=(12, 3))
    plt.subplot(131)
    plt.title('Predict Confusion Matrix')
    sn.heatmap(cm, annot=True, fmt='.10g', cmap=plt.cm.Blues)
    plt.xlabel('Predict label')
    plt.ylabel('True label')

    plt.subplot(132)
    precision_cm = np.copy(cm)
    precision_cm = precision_cm / precision_cm.sum(axis=0)
    plt.title('Precision Confusion Matrix')
    sn.heatmap(precision_cm, annot=True, fmt='.10g', cmap=plt.cm.Blues)
    plt.xlabel('Predict label')
    plt.ylabel('True label')

    plt.subplot(133)
    recall_cm = np.copy(cm)
    recall_cm = recall_cm / recall_cm.sum(axis=1).reshape((2, 1))
    plt.title('Recall Confusion Matrix')
    sn.heatmap(recall_cm, annot=True, fmt='.10g', cmap=plt.cm.Blues)
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.show()
    return model, history


maxlen = 30
max_words = 3000
train_x, train_y, test_x, test_y = preprocess_data(all_df, maxlen=maxlen, max_words=max_words)
model = build_model1(max_words, maxlen)
epochs = 20
batch_size = 32
model, history = train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=epochs, batch_size=batch_size)

# Model overfitting is serious, try to use regularization for optimization


def build_model2(max_words, maxlen):
    model = Sequential()
    model.add(Embedding(max_words, 64, input_length=maxlen))
    model.add(Flatten(name='feature'))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model


model = build_model2(max_words, maxlen)
epochs = 20
batch_size = 32
model, history = train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=epochs, batch_size=batch_size)

# In recent years, recurrent neural networks have achieved excellent results in
# text data processing, so we decided to use recurrent neural networks to try
# to improve the model effect


def build_model3(max_words, maxlen):
    model = Sequential()
    model.add(Embedding(max_words, 64, input_length=maxlen))
    model.add(LSTM(64, name='feature'))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.05)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model


model = build_model3(max_words, maxlen)
epochs = 20
batch_size = 32
model, history = train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=epochs, batch_size=batch_size)


# Add more LSTM laysers
def build_model4(max_words, maxlen):
    model = Sequential()
    model.add(Embedding(max_words, 64, input_length=maxlen))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32, name='feature'))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.05)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model


model = build_model4(max_words, maxlen)
epochs = 20
batch_size = 32
model, history = train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=epochs, batch_size=batch_size)

# Use Bidirectional RNN
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras import regularizers


def build_model5(max_words, maxlen):
    model = Sequential()
    model.add(Embedding(max_words, 64, input_length=maxlen))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(32), name='feature'))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.05)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model


model = build_model5(max_words, maxlen)
epochs = 20
batch_size = 32
model, history = train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=epochs, batch_size=batch_size)

# Bidirectional is not usefull , and still overfitting, try to use dropout for optimization
from tensorflow.keras.layers import Dropout


def build_model6(max_words, maxlen):
    model = Sequential()
    model.add(Embedding(max_words, 64, input_length=maxlen))
    model.add(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
    model.add(LSTM(32, dropout=0.3, recurrent_dropout=0.3, name='feature'))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.05)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model


model = build_model6(max_words, maxlen)
epochs = 20
batch_size = 32
model, history = train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=epochs, batch_size=batch_size)

"""Analyzing the above experimental data, we found that the model effect is very unstable, considering that we have less sample data, only 3000 items,  training Embedding layer from zero may not achieve better results, so we consider using BERT(a transformer encoder architecture) to improve the model effect"""

# !pip install tensorflow-text
# !pip install tf-models-official

import tensorflow as tf
import tensorflow_hub as hub

tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'


# process raw data into train_x, train_y, test_x, test_y
def preprocess_data(all_df, test_split=0.2):
    test_df = all_df.sample(frac=test_split, random_state=6)
    train_df = all_df.append(test_df).drop_duplicates(keep=False)
    print(test_df.shape)
    print(train_df.shape)

    train_x = train_df['text'].values
    train_y = train_df['label'].astype(float).values

    test_x = test_df['text'].values
    test_y = test_df['label'].astype(float).values
    return train_x, train_y, test_x, test_y


def build_model7():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = Dropout(0.1, name='feature')(net)
    net = Dense(1, activation='sigmoid', name='classifier')(net)
    model = tf.keras.Model(text_input, net)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model


train_x, train_y, test_x, test_y = preprocess_data(all_df)
model = build_model7()
epochs = 20
batch_size = 32
model, history = train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=epochs, batch_size=batch_size)


# Use RNN with bert
def build_model8():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['sequence_output']
    net = LSTM(64, return_sequences=True)(net)
    net = LSTM(32, name='feature')(net)
    net = Dense(64, activation='relu')(net)
    net = Dropout(0.5)(net)
    net = Dense(1, activation='sigmoid', name='classifier')(net)
    model = tf.keras.Model(text_input, net)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model


model = build_model8()
epochs = 20
batch_size = 32
model, history = train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=epochs, batch_size=batch_size)

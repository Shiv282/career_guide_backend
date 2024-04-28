#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import string
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding, Bidirectional


# In[ ]:





# In[2]:


# Text preprocessing function
def clean(text):
    global str_punc
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text    

str_punc = string.punctuation.replace(',', '').replace("'",'')


# In[3]:


# Read dataset & Preprocess text
dataset = pd.read_csv('./dataset.csv')

X = dataset['Sentence'].apply(clean)
y = dataset['Type']


# In[4]:


# Visualize classes counts
dataset['Type'].hist()


# In[5]:


# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)


text_train, text_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Tokenize words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_train)
sequences_train = tokenizer.texts_to_sequences(text_train)
sequences_test = tokenizer.texts_to_sequences(text_test)
X_train = pad_sequences(sequences_train, maxlen=48, truncating='pre')
X_test = pad_sequences(sequences_test, maxlen=48, truncating='pre')

vocabSize = len(tokenizer.index_word) + 1
print(f"Vocabulary Size = {vocabSize}")


# In[6]:


# Read GloVE embeddings

path_to_glove_file = './glove.txt'
num_tokens = vocabSize
embedding_dim = 200
hits = 0
misses = 0
embeddings_index = {}

# Read word vectors
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs
print("Found %s word vectors." % len(embeddings_index))


# Assign word vectors to our dictionary/vocabulary
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i][:100] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))


# In[7]:


# Build neural network architecture

model = Sequential()
model.add(Embedding(vocabSize, 200, input_length=X_train.shape[1], weights=[embedding_matrix], trainable=False))
model.add(LSTM(256, dropout=0.2,recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(128, dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[8]:


history = model.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    verbose=1,
                    batch_size=64,
                    epochs=10)


# In[9]:


# Visualize Loss & Accuracy




# In[10]:


model.evaluate(X_test, y_test, verbose=1)


# In[11]:


# Classify custom sample

sentences = ["Can you imagine how this idea's gonna change the education system!", # Visual
             "Brilliant! I can't wait to hear the news about this change!", # Auditory
             "Chill out guys, nothing's gonna change, we have to study hard to succeed" # Kinesthetic
            ]
for sentence in sentences:
    print(sentence)
    sentence = clean(sentence)
    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=48, truncating='pre')
    result = le.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
    proba =  np.max(model.predict(sentence))
    print(f"{result} : {proba}\n\n")


# In[13]:


import pickle

with open('tokenizer_new.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)
    
with open('labelEncoder_new.pickle', 'wb') as f:
    pickle.dump(le, f)
    
    
model.save('LearningStyleClassifier_new.h5')


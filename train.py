from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras
import numpy as np
import os
from sklearn.model_selection import train_test_split

embeddings = np.load('X.npy', allow_pickle=True)
moves = np.load('Y.npy', allow_pickle=True)

X_train, X_val, y_train, y_val = train_test_split(embeddings, moves, test_size=0.1, random_state=123)

print(embeddings.shape)
print(moves.shape)

# Define the RNN-based language model
if os.path.isdir('model'):
    model = keras.models.load_model('model')
else:
    model: Sequential = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(embeddings.shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(moves.shape[1], activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam')

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

# Train the language model
model.fit(X_train, y_train, epochs=25, batch_size=30, validation_data=(X_val, y_val), callbacks=[early_stopping])

model.save('model')
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical

pixel_train = pd.read_csv('mnist-train.csv')
pixel_test = pd.read_csv('mnist-test.csv')
pixel_train = pixel_train.to_numpy()
pixel_test = pixel_test.to_numpy()

X_train = np.zeros((1000,784))
y_train = np.zeros(1000)

for i in range(1000):
    X_train[i] = pixel_train[i,1:]
    y_train[i] = pixel_train[i,0]

X_test = np.zeros((500,784))
y_test = np.zeros(500)

for i in range(500):
    X_test[i] = pixel_test[i,1:]
    y_test[i] = pixel_test[i,0]

X_train = X_train.reshape(1000,28,28,1)
X_test = X_test.reshape(500,28,28,1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='category_crossentropy', metrix=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
pred_y = model.predict(X_test)
error = np.mean(pred_y != y_test)
print(error)

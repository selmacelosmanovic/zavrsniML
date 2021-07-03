import numpy as np
import pandas as pd
from keras import layers
from keras import models
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('C:/Users/selma/OneDrive/Desktop/data/dataset.csv', sep=',')
y = data.points
# print(y)

X = data.copy(deep=True)
X.drop(columns=['points'], inplace=True)
# print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(np.array(X_train).shape)
model = models.Sequential()

model.add(layers.Dense(64, activation='relu', name='dense_1', input_shape=(6,)))
model.add(layers.Dense(64, activation='relu', name='dense_2'))
model.add(layers.Dense(1, name='output'))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, Y_train, epochs=100, batch_size=1, validation_split=0.1).history
mae = history['mae']
val_mae = history['val_mae']

loss_values = history['loss']
val_loss_values = history['val_loss']

print("MAE je: ", mae)
print("Gubitak je: ", loss_values)

epochs = range(1, len(mae) + 1)

plt.plot(epochs, loss_values, 'yo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

results = model.evaluate(X_test, Y_test)  # results su test_loss i test_acc
print("Rezultati: ", results)

model.save('C:/Users/selma/OneDrive/Desktop/ETF/3. Godina/6. Semestar/Završni rad/model')



import numpy as np
from tensorflow import keras

model = keras.models.load_model('C:/Users/selma/OneDrive/Desktop/ETF/3. Godina/6. Semestar/Završni rad/model')

arr = np.array([0.04, 0.01, 0.35, 0, 0, 0.001])
prediction = model.predict(arr.reshape(1, 6))
print("Bodovi za ovaj projekat su: ", prediction[0][0])

import numpy as np
from tensorflow import keras
import pandas as pd

pathModel = input("Unesite putanju do direktorija sa modelom: ")
model = keras.models.load_model(pathModel)  # putanja do modela
mae = keras.losses.MeanAbsoluteError()

path = input("Unesite putanju do direktorija sa projektom: ")
path = path + '/dataFinal.csv'
data = pd.read_csv(path, sep=',')
arr = data.to_numpy()
prediction = model.predict(arr.reshape(1, 6))

print("Bodovi za ovaj projekat su: ", prediction[0][0])

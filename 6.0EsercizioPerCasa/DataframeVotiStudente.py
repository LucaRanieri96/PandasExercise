import pandas as pd
from numpy import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

idStudente = []

for i in range (1000):
    idStudente.append(i)
#print(idStudente)

votoItaliano = random.randint(1,11, size=(1000))
votoStoria = random.randint(1,11, size=(1000))
votoGeografia= random.randint(1,11, size=(1000))
votoMatematica= random.randint(1,11, size=(1000))
votoFisica= random.randint(1,11, size=(1000))
votoInglese= random.randint(1,11, size=(1000))

dictVoti = {
"idStudente": idStudente,
"votoItaliano": votoItaliano,
"votoStoria": votoStoria,
"votoGeografia": votoGeografia,
"votoMatematica": votoMatematica,
"votoFisica": votoFisica,
"votoInglese": votoInglese
}

#print(dataframe)

df= pd.DataFrame(dictVoti)
# print("\nDataFrame creato da un dizionario:")
# print(df.head())

#varianza max min var media
#print(df.describe()) printa all 

min = df.min()
max = df.max()
std = df.std()
mean = df.mean()
#print(min, max, std, mean)
#print(f'In italiano il voto medio è {df["votoItaliano"].mean()}, la varianza è {df["votoItaliano"].std()}, il min è {df["votoItaliano"].min()}, il max è {df["votoItaliano"].max()}')

#.⁠ ⁠Fare preprocessing e normalizzare i voti con min max scaler e togliere l'id studente

df = df.drop(columns=['idStudente'])

normalizer = MinMaxScaler()

df['votoItaliano'] = normalizer.fit_transform(df[['votoItaliano']])
df['votoStoria'] = normalizer.fit_transform(df[['votoStoria']])
df['votoGeografia'] = normalizer.fit_transform(df[['votoGeografia']])
df['votoMatematica'] = normalizer.fit_transform(df[['votoMatematica']])
df['votoFisica'] = normalizer.fit_transform(df[['votoFisica']])
df['votoInglese'] = normalizer.fit_transform(df[['votoInglese']])

#crea una colonna media voti alunno
df['media'] = df[['votoItaliano', 'votoStoria', 'votoGeografia','votoMatematica','votoFisica','votoInglese']].mean(axis=1)

#Crea una colonna per le labels cioè se uno studente è promosso o no: df['promosso'] = df['Media Voto'] >= 6
df['promosso'] = df['media'] >= 0.6

print(df.head(20))
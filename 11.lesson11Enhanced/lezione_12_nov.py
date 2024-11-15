import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n_studenti = 1000

dataframe = pd.DataFrame()

dataframe['id_studente'] = pd.Series(range(1, n_studenti +1))

materie = ['voti_italiano', 'voti_storia','voti_geografia','voti_matematica','voti_fisica','voti_inglese']

for materia in materie:
    voti = []
    for i in range(n_studenti):
        voti.append(np.random.randint(1,11))
    dataframe[materia] = pd.Series(voti)

#print(dataframe.describe()) - per tutte le statistiche 

#altro metodo per selezione delle statistiche
statistiche = dataframe.agg(['mean','min', 'max', 'std'])
print(statistiche)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('/drug200.csv')

print(df.info())
print(df.describe())
print(df.isnull().sum()) 


# Inizializzazione degli encoder e scaler
lab_encoder = LabelEncoder()
mm_scaler = MinMaxScaler()

# Lista delle feature categoriche
featureStringate = ['Sex', 'BP', 'Cholesterol']

# Codifica delle feature categoriche
for feature in featureStringate:
    df[feature] = lab_encoder.fit_transform(df[feature])

# Codifica della colonna target 'Drug'
df['Drug'] = lab_encoder.fit_transform(df['Drug'])

# Separa la colonna target
y = df['Drug']

# Applica lo scaling solo sulle feature (escludendo 'Drug')
features = df.drop(columns='Drug')
features_scaled = pd.DataFrame(mm_scaler.fit_transform(features), columns=features.columns)

# Definizione delle variabili X (feature) e y (target)
X = features_scaled

# Suddivisione del dataset in training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creazione e addestramento del modello Decision Tree
model_dtc = DecisionTreeClassifier()
model_dtc.fit(X_train, y_train)

# Verifica dell'addestramento
print("Modello addestrato con successo!")
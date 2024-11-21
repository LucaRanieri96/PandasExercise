# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data_raw = pd.read_csv('wine_quality_last.csv')
print(data_raw.head(20))

# %%
print(data_raw.describe())

print(data_raw.info())
print()


print("Valori nulli")
print(data_raw.isna().sum())
print()

data_raw = data_raw.dropna()
print(data_raw.isna().sum())


print("Numero di duplicati")
print(data_raw.duplicated().sum())
data_raw = data_raw.drop_duplicates()

print(data_raw.duplicated().sum())
print()

print(data_raw.head())

data_raw = data_raw.dropna()
data_raw = data_raw.drop_duplicates()

print("\nDataFrame dopo aver eliminato righe nulli e duplicati:")
print(data_raw)
print(data_raw.columns)
statistica = pd.DataFrame()
volatile_acidity_mean = data_raw['volatile acidity'].mean()
fixed_acidity_mean = data_raw['fixed acidity'].mean()
citric_acid_mean = data_raw['citric acid'].mean()
sulphates_mean = data_raw["sulphates"].mean()
alcohol_mean = data_raw['alcohol'].mean()
quality_mean = data_raw['quality'].mean()
residual_sugar_mean = data_raw["residual sugar"].mean()
chlorides_mean = data_raw["chlorides"].mean()
free_sulfur_dioxide_mean = data_raw["free sulfur dioxide"].mean()
density_mean = data_raw["density"].mean()
ph_mean = data_raw["pH"].mean()
total_sulfur_dioxide_mean = data_raw["total sulfur dioxide"].mean()

volatile_acidity_median = data_raw['volatile acidity'].median()
fixed_acidity_median = data_raw['fixed acidity'].median()
citric_acid_median = data_raw['citric acid'].median()
sulphates_median = data_raw['sulphates'].median()
alcohol_median = data_raw['alcohol'].median()
quality_median = data_raw['quality'].median()
residual_sugar_median = data_raw["residual sugar"].median()
chlorides_median = data_raw["chlorides"].median()
free_sulfur_dioxide_median = data_raw["free sulfur dioxide"].median()
density_median = data_raw["density"].median()
ph_median = data_raw["pH"].median()
total_sulfur_dioxide_median = data_raw["total sulfur dioxide"].median()

volatile_acidity_std = data_raw['volatile acidity'].std()
fixed_acidity_std = data_raw['fixed acidity'].std()
citric_acid_std = data_raw['citric acid'].std()
sulphate_std = data_raw['sulphates'].std()
alcohol_std = data_raw['alcohol'].std()
quality_std = data_raw['quality'].std()
residual_sugar_std = data_raw["residual sugar"].std()
chlorides_std = data_raw["chlorides"].std()
free_sulfur_dioxide_std = data_raw["free sulfur dioxide"].std()
density_std = data_raw["density"].std()
ph_std = data_raw["pH"].std()
total_sulfur_dioxide_std = data_raw["total sulfur dioxide"].std()

statistica['metric'] = ["mean", "median", "std"]
statistica['volatile acidity'] = [volatile_acidity_mean, volatile_acidity_median, volatile_acidity_std]
statistica['citric acid'] = [citric_acid_mean, citric_acid_median, citric_acid_std]
statistica['fixed acidity'] = [fixed_acidity_mean, fixed_acidity_median, fixed_acidity_std]
statistica["sulphate"] = [sulphates_mean, sulphates_median, sulphate_std]
statistica["alcohol"]=[alcohol_mean,alcohol_median,alcohol_std]
statistica["quality"]=[quality_mean,quality_median,quality_std]
statistica["residual sugar"] =[residual_sugar_mean, residual_sugar_median,residual_sugar_std]
statistica["chlorides"] = [chlorides_mean, chlorides_median, chlorides_std]
statistica["free sulfur dioxide"] = [free_sulfur_dioxide_mean, free_sulfur_dioxide_median, free_sulfur_dioxide_std]
statistica["density"] = [density_mean, density_median, density_std]
statistica["pH"] = [ph_mean, ph_median, ph_std]
statistica["total sulfur dioxide"] = [total_sulfur_dioxide_median, total_sulfur_dioxide_mean, total_sulfur_dioxide_std]

print(statistica.head())

# %%
normalizer = MinMaxScaler()
label_encoder = LabelEncoder()

data_raw["volatile acidity"] = normalizer.fit_transform(data_raw[["volatile acidity"]])
data_raw['citric acid'] = normalizer.fit_transform(data_raw[['citric acid']])
data_raw["fixed acidity"] = normalizer.fit_transform(data_raw[["fixed acidity"]])
data_raw['sulphates'] = normalizer.fit_transform(data_raw[["sulphates"]])
data_raw["alcohol"] = normalizer.fit_transform(data_raw[["alcohol"]])
data_raw['quality'] = normalizer.fit_transform(data_raw[['quality']])
data_raw["residual sugar"] = normalizer.fit_transform(data_raw[["residual sugar"]])
data_raw['chlorides'] = normalizer.fit_transform(data_raw[["chlorides"]])
data_raw["free sulfur dioxide"] = normalizer.fit_transform(data_raw[["free sulfur dioxide"]])
data_raw['density'] = normalizer.fit_transform(data_raw[['density']])
data_raw["pH"] = normalizer.fit_transform(data_raw[["pH"]])
data_raw['total sulfur dioxide'] = normalizer.fit_transform(data_raw[["total sulfur dioxide"]])
# %%
label_encoder = LabelEncoder()
label_encoder.fit(data_raw[['quality']])
data_raw['quality'] = label_encoder.transform(data_raw[['quality']])
# %%
X = data_raw[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
y = data_raw['quality']
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size = 0.2)
# %%
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
# %%
y_pred = svm.predict(X_test)
print(X_test)
# %%
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuratezza: {accuracy:.2f}')
# %%

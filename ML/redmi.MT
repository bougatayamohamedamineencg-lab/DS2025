# 🍷 TP – Classification de la Qualité des Vins  
## Dataset : Wine Quality (UCI) – Version ucimlrepo  
## Modèle : k-NN + Normalisation

---

## 🔹 Installation et Importation

```python
pip install ucimlrepo
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
# Chargement du dataset Wine Quality (id = 186)
data = fetch_ucirepo(id=186)

# Séparation X (features) et y (target)
X = data.data.features
y = data.data.targets["quality"]

print(X.head())
print(X.info())
print(y.value_counts())
# 0 = vin de mauvaise qualité (<=5)
# 1 = vin de bonne qualité (>=6)

Y = np.array([0 if q <= 5 else 1 for q in y])

unique, counts = np.unique(Y, return_counts=True)
print(dict(zip(unique, counts)))
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,5))
sns.boxplot(data=X)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(10,8))
sns.heatmap(X.corr(), annot=False, cmap="coolwarm")
plt.show()
from sklearn.model_selection import train_test_split

# 1/3 test
Xa, Xt, Ya, Yt = train_test_split(X, Y, test_size=1/3, stratify=Y, shuffle=True)

# 1/2 validation
Xa, Xv, Ya, Yv = train_test_split(Xa, Ya, test_size=0.5, stratify=Ya, shuffle=True)

print(len(Xa), len(Xv), len(Xt))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k = 3
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(Xa, Ya)

Ypred_v = clf.predict(Xv)
error_v = 1 - accuracy_score(Yv, Ypred_v)

print("Erreur validation (k=3) :", error_v)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k = 3
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(Xa, Ya)

Ypred_v = clf.predict(Xv)
error_v = 1 - accuracy_score(Yv, Ypred_v)

print("Erreur validation (k=3) :", error_v)
k_vector = np.arange(1, 40, 2)
error_train = []
error_val = []

for k in k_vector:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(Xa, Ya)

    Ypred_train = clf.predict(Xa)
    Ypred_val = clf.predict(Xv)

    error_train.append(1 - accuracy_score(Ya, Ypred_train))
    error_val.append(1 - accuracy_score(Yv, Ypred_val))
plt.figure(figsize=(10,5))
plt.plot(k_vector, error_train, label="Erreur d'entraînement")
plt.plot(k_vector, error_val, label="Erreur de validation")
plt.xlabel("k")
plt.ylabel("Erreur")
plt.legend()
plt.show()
best_index = np.argmin(error_val)
k_star = k_vector[best_index]

print("k optimal :", k_star)
clf_best = KNeighborsClassifier(n_neighbors=k_star)
clf_best.fit(Xa, Ya)

Ypred_test = clf_best.predict(Xt)
error_test = 1 - accuracy_score(Yt, Ypred_test)

print("Erreur test avec k optimal :", error_test)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(Xa)

Xa_n = sc.transform(Xa)
Xv_n = sc.transform(Xv)
Xt_n = sc.transform(Xt)
error_train_n = []
error_val_n = []

for k in k_vector:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(Xa_n, Ya)

    Ypred_train = clf.predict(Xa_n)
    Ypred_val = clf.predict(Xv_n)

    error_train_n.append(1 - accuracy_score(Ya, Ypred_train))
    error_val_n.append(1 - accuracy_score(Yv, Ypred_val))
plt.figure(figsize=(10,5))
plt.plot(k_vector, error_train_n, label="Erreur train (normalisé)")
plt.plot(k_vector, error_val_n, label="Erreur validation (normalisé)")
plt.xlabel("k")
plt.ylabel("Erreur")
plt.legend()
plt.show()

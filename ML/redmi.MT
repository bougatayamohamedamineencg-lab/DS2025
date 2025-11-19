# 📘 Compte Rendu – Classification de la Qualité des Vins  
*(Wine Quality Dataset – k-NN & Normalisation)*

## 1. Introduction
L’objectif de ce TP est de construire un modèle de classification permettant de prédire la **qualité du vin blanc** à partir de mesures physico-chimiques.  
Le dataset provient de l’UCI Machine Learning Repository et contient **4898 échantillons** et **11 variables explicatives**.

Le travail comprend :
- Analyse exploratoire des données  
- Préparation de la cible  
- Split entraînement / validation / test  
- Modélisation via **k-NN**  
- Sélection du meilleur k  
- Impact de la normalisation des données  

---

## 2. Chargement du Dataset et Exploration
Le dataset est chargé via `ucimlrepo`, ce qui évite l'utilisation des liens UCI manuels.

L’analyse préliminaire montre :
- 4898 observations  
- 11 variables d’entrée (acidité, pH, alcool, densité…)  
- Une variable cible : *quality*  
- Répartition des notes entre 3 et 9

---

## 3. Transformation en Problème Binaire
Pour simplifier :

- **0 → Mauvaise qualité** (quality ≤ 5)  
- **1 → Bonne qualité** (quality ≥ 6)

Légère dominance des bons vins dans les données.

---

## 4. Analyse Statistique

### 4.1 Boxplots  
- Présence d’outliers  
- Variables sur des échelles très différentes  

### 4.2 Heatmap de corrélation  
- Forte corrélation densité ↔ sucre résiduel  
- Corrélation négative alcool ↔ densité  

→ Importance de normaliser avant k-NN.

---

## 5. Split des Données
Division en 3 ensembles (stratifiés) :

- **Entraînement : 33%**  
- **Validation : 33%**  
- **Test : 33%**

La stratification garantit une répartition équivalente des classes.

---

## 6. Modèle k-Nearest Neighbors (k-NN)

### 6.1 Test avec k = 3
Taux d’erreur sur validation obtenu comme baseline.

### 6.2 Recherche du meilleur k
Tests pour k ∈ [1, 40] :

- petits k : overfitting  
- grands k : underfitting  

Le **k optimal** correspond au minimum d’erreur validation.

### 6.3 Évaluation finale
Le meilleur modèle généralise bien sur le jeu de test.

---

## 7. Normalisation
Normalisation avec `StandardScaler` :

- Ajusté uniquement sur le training set  
- Appliqué à validation et test  
- Évite la fuite d’information  

Résultats :

- courbes plus stables  
- meilleures performances  
- k optimal différent  

---

## 8. Conclusion

- k-NN nécessite une normalisation pour de bonnes performances.  
- Sélection correcte de k via validation.  
- Dataset adapté pour un cas réel de classification.  
- Le modèle final normalisé présente une bonne généralisation.

Fin du compte rendu.


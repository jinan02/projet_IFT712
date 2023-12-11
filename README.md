
Projet IFT712 : Classification de Feuilles (Leaf Classificiation)
============================================================

* Free software: MIT license




## Aperçu

ce projet vise à résoudre le défi de classification des feuilles en exploitant six modèles distincts de classification. Chaque modèle est élaboré afin de prédire la catégorie d'une feuille en se fondant sur des caractéristiques spécifiques extraites.

## Structure du Projet

    ├── data/
    │   ├── train.csv
    │   ... 
    ├── src/
    │   ├── __init__.py
    │   ├── Classify_Data.py
    │   ├── Preprocess_Data.py
    │   ├── Visualize_Results.py
    │   ... 
    ├── requirements.txt
    ├── projet.ipynb
    ├── LICENSE
    ├── AUTHORS.rst
    ├── .editorconfig
    ├── Makefile
    ├── .travis.yml
    ├── .gitignore
    └── README.md

La structure du projet est organisée comme suit :

- **data/** : Répertoire contenant les données du projet, avec notamment le fichier `train.csv`.

- **src/** : Répertoire abritant les scripts source du projet :
  - **_init_.py** : Fichier d'initialisation du module Python.
  - **Classify_Data.py** : Script pour la classification des données.
  - **Preprocess_Data.py** : Script pour le prétraitement des données.
  - **Visualize_Results.py** : Script pour la visualisation des résultats.
  - ...

- **requirements.txt** : Fichier spécifiant les dépendances du projet.

- **projet.ipynb** : Fichier Jupyter Notebook contenant le code principal du projet.

- **LICENSE** : Fichier décrivant la licence du projet.

- **AUTHORS.rst** : Fichier listant les auteurs du projet.

- **.editorconfig** : Fichier de configuration pour les éditeurs de texte.

- **Makefile** : Fichier utilisé pour automatiser des tâches dans le projet.

- **.travis.yml** : Fichier de configuration pour l'intégration continue avec Travis CI.

- **.gitignore** : Fichier spécifiant les fichiers et répertoires à ignorer lors du suivi avec Git.

- **README.md** : Fichier que vous êtes actuellement en train de lire, contenant des informations générales sur le projet.


## Données

Les données destinées à l'entraînement et aux tests sont consignées dans le répertoire `data`. Le corpus de données en question compte approximativement 991 entrées.

Les colonnes contenues dans le jeu de données sont définies comme suit :
- `id` : un identifiant anonyme unique pour une image
- `margin_1`, `margin_2`, ..., `margin_64` : chacun des 64 vecteurs d'attributs pour la caractéristique de marge
- `shape_1`, `shape_2`, ..., `shape_64` : chacun des 64 vecteurs d'attributs pour la caractéristique de forme
- `texture_1`, `texture_2`, ..., `texture_64` : chacun des 64 vecteurs d'attributs pour la caractéristique de texture

Pour plus de détails sur les données, veuillez consulter la [source des données sur Kaggle](https://www.kaggle.com/c/leaf-classification/data).

## Modèles de Classification

1. **Modèle 1 - Régression Logistique :** La régression logistique est un modèle de classification utilisé pour prédire la probabilité qu'une observation appartienne à une catégorie particulière. Elle fonctionne en ajustant une courbe logistique aux données d'entraînement. [Documentation de la Régression Logistique](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

2. **Modèle 2 - Support Vector Classification (SVC) :** Le modèle de classification par machine à vecteurs de support (SVC) cherche à séparer les classes en trouvant le meilleur hyperplan qui maximise la marge entre les différentes classes. [Documentation du SVC](https://scikit-learn.org/stable/modules/svm.html#classification)

3. **Modèle 3 - SGD (Stochastic Gradient Descent) :** Le SGD est un algorithme d'optimisation utilisé dans le contexte de la classification. Il ajuste les paramètres du modèle en utilisant la descente de gradient sur des échantillons aléatoires. [Documentation du SGD](https://scikit-learn.org/stable/modules/sgd.html)

4. **Modèle 4 - Random Forest :** Random Forest est un ensemble d'arbres de décision où chaque arbre vote pour une classe et la classe majoritaire est choisie. Cela améliore la robustesse et la précision du modèle. [Documentation du Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)

5. **Modèle 5 - Adaboost :** Adaboost est un modèle de boosting qui combine plusieurs modèles faibles pour former un modèle fort. Il assigne des poids aux observations, donnant plus d'importance aux erreurs précédentes. [Documentation de l'Adaboost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)

6. **Modèle 6 - Bagging :** Le bagging (Bootstrap Aggregating) est une technique d'ensemble où plusieurs modèles sont entraînés sur différents sous-ensembles des données d'entraînement. Cela réduit la variance du modèle global. [Documentation du Bagging](https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator)

## Métriques de Performance

L'évaluation de la performance des modèles de classification repose sur plusieurs métriques clés. Voici un aperçu des principales métriques utilisées dans ce projet :

1. **Accuracy (Précision) :**
   - **Définition :** L'accuracy mesure la proportion d'observations correctement classées parmi l'ensemble des observations.
   - **Formule :** \(\frac{\text{Nombre d'observations correctement classées}}{\text{Nombre total d'observations}}\)
   - **Interprétation :** Une valeur élevée d'accuracy indique une performance globale du modèle.

2. **Precision (Précision) :**
   - **Définition :** La précision mesure la proportion d'observations positives correctement classées parmi toutes les observations classées comme positives.
   - **Formule :** \(\frac{\text{Nombre de vrais positifs}}{\text{Nombre de vrais positifs + Nombre de faux positifs}}\)
   - **Interprétation :** Une valeur élevée de précision indique que le modèle minimise les faux positifs.

3. **Recall (Rappel) :**
   - **Définition :** Le rappel mesure la proportion d'observations positives correctement classées parmi toutes les observations réellement positives.
   - **Formule :** \(\frac{\text{Nombre de vrais positifs}}{\text{Nombre de vrais positifs + Nombre de faux négatifs}}\)
   - **Interprétation :** Un rappel élevé indique que le modèle capture efficacement les observations positives réelles.

4. **F1 Score :**
   - **Définition :** Le F1 score est la moyenne harmonique de la précision et du rappel, fournissant une mesure équilibrée entre les deux.
   - **Formule :** \(2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision + Recall}}\)
   - **Interprétation :** Un F1 score élevé indique un équilibre optimal entre la précision et le rappel.


## Installation

   ```bash
   # Clonez le dépôt GitHub
   git clone [https://github.com/jinan02/projet_ift712](https://github.com/jinan02/projet_ift712)
   cd nom-du-projet
   
   # Créez un environnement virtuel (optionnel, mais recommandé)
   python -m venv venv

   # Pour activer l'environnement virtuel (sous Windows) 
   venv\Scripts\activate

   # Pour activer l'environnement virtuel (sous macOS/Linux)
   source venv/bin/activate

   #Installez les dépendances
   pip install -r requirements.txt

```

## Utilisation
Une fois que l'installation est terminée, vous pouvez lancer le projet en suivant ces étapes :
Pour entraîner et utiliser les modèles de classification, exécutez le script principal projet.ipynb. Vous pouvez utiliser un environnement Jupyter Notebook ou tout autre environnement de développement Python de votre choix.

```bash
# Exemple de commande pour exécuter le script d'entraînement
jupyter notebook projet.ipynb
```

## Collaboration

Nous sommes reconnaissants envers les contributeurs qui ont déjà participé à ce projet. Voici une liste des contributeurs actuels :

1. **Abir Jamaly**
2. **Jinane Boufaris**
3. **Hamza Boussairi**
   
Merci à eux pour leurs efforts et leur engagement envers l'amélioration continue de ce projet.
Nous encourageons vivement la collaboration et les contributions à ce projet. Si vous souhaitez participer, voici quelques lignes directrices pour une collaboration harmonieuse :

### Comment Contribuer

1. **Créez une Issue :** Avant de commencer à contribuer, créez une issue pour discuter des modifications que vous envisagez. Cela permet d'obtenir des commentaires et d'éviter tout chevauchement avec d'autres contributeurs.

2. **Fork et Clone :** Fork le dépôt vers votre propre compte GitHub et clonez-le sur votre machine locale.

   ```bash
   git clone https://github.com/jinan02/projet_ift712
   ```

3. **Créez une Branche :** Créez une branche pour travailler sur votre nouvelle fonctionnalité ou correction.

   ```bash
   git checkout -b nom-de-votre-branche
   ```

4. **Effectuez les Modifications :** Faites vos modifications et assurez-vous de suivre les conventions de codage existantes.

5. **Testez Localement :** Avant de soumettre une demande de fusion, assurez-vous que votre code fonctionne correctement localement.

6. **Soumettez une Pull Request :** Lorsque vous êtes prêt, soumettez une pull request (PR) depuis votre branche vers le dépôt principal. Expliquez clairement les changements que vous avez apportés et pourquoi ils sont nécessaires.

### Normes de Contribution

- Suivez les normes de codage du projet. Consultez le fichier `.editorconfig` pour les configurations d'éditeur recommandées.
- Documentez tout nouveau code ou modification apportée.
- Ajoutez des tests si nécessaire pour garantir la stabilité du code.
- Respectez les règles de versionnement sémantique pour les changements dans le code.

### Problèmes et Demandes d'Améliorations

Si vous identifiez des problèmes ou si vous avez des suggestions d'améliorations, n'hésitez pas à créer une issue. Nous sommes ouverts à l'amélioration continue de ce projet.

Nous vous remercions pour votre intérêt et votre contribution à faire de ce projet un succès collaboratif !





Authors
--------

- Abir Jamaly
- Hamza Boussairi
- Jinane Boufaris


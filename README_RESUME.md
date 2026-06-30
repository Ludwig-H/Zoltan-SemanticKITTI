# Instructions pour la reprise du travail sur le nouveau Codespace

Ce fichier contient toutes les informations et instructions nécessaires pour configurer le nouveau Codespace plus puissant, installer les dépendances et reprendre la génération et la visualisation interactives.

## 1. Contexte du projet
Nous avons créé un tableau de bord interactif WebGL (Three.js) dans [visualize_hgp.html](file:///workspaces/Zoltan-SemanticKITTI/visualize_hgp.html) pour visualiser le clustering par percolation d'hypergraphes (HGP) sur les nuages de points SemanticKITTI.
L'interface permet de sélectionner de manière interactive :
- **La Séquence** (`00`, `08`)
- **La Trame** (`0` à `4`)
- **L'ordre K** (`1` à `5`)
- **L'exposant expZ** (`1.0`, `1.5`, `2.0`, `2.5`, `3.0`)

Afin de garantir des transitions instantanées dans le navigateur, les données géométriques stables d'un côté (positions 3D, vérité terrain, classes) et les labels de clustering HGP de l'autre sont pré-calculés par le script `src/generate_visualizations.py` et sauvegardés sous forme de fichiers JSON plats dans le dossier (ignoré par git) `visualize/`.

---

## 2. Configuration du nouveau Codespace

### Étape 2.1 : Installer les dépendances système C++
Le package `HGP-clusterer` compile des extensions C++ et Cython (notamment pour l'ordre-k Delaunay 3D). Installez les paquets requis :
```bash
sudo apt-get update
sudo apt-get install -y libcgal-dev libeigen3-dev cmake build-essential
```

### Étape 2.2 : Cloner et compiler le package `HGP-clusterer`
Si le répertoire `HGP-clusterer` n'est pas déjà présent ou cloné dans le workspace :
```bash
git clone https://github.com/Ludwig-H/HGP-clusterer.git
```
Installez ensuite le package en mode éditable avec ses dépendances Python (incluant Gudhi, Scikit-learn, UMAP, etc.) :
```bash
cd HGP-clusterer
pip install -e .[umap]
cd ..
```
*Note : La compilation du binding Cython et C++ se fait automatiquement à l'installation.*

---

## 3. Lancement de la génération des visualisations

Le script principal est [src/generate_visualizations.py](file:///workspaces/Zoltan-SemanticKITTI/src/generate_visualizations.py).
Il est configuré pour calculer la grille multi-paramètres complète (150 combinaisons au total).

### Choix du backend géométrique :
Par défaut, le script utilise `backend='geogram'` qui supporte le multithreading et est extrêmement rapide.
* **Option A (Recommandée - Rapide) :** Si Geogram s'est compilé correctement (via le dossier `geogram_install` s'il a été récupéré), lancez simplement le script.
* **Option B (Alternative CGAL) :** Si Geogram n'est pas disponible, modifiez la ligne `backend='geogram'` en `backend='cgal'` dans le script [src/generate_visualizations.py](file:///workspaces/Zoltan-SemanticKITTI/src/generate_visualizations.py). CGAL est inclus par défaut avec `libcgal-dev` et est ultra-robuste.

### Lancer la génération :
Pour lancer la génération en tâche de fond et suivre l'avancement :
```bash
python src/generate_visualizations.py
```
*(Le script de génération a été optimisé par monkeypatching pour mettre en cache les triangulations Delaunay d'ordre K par trame, évitant les calculs redondants pour les différentes valeurs d'expZ et multipliant la vitesse par 3).*

---

## 4. Visualisation dans le navigateur

Pendant que les calculs s'exécutent (ou une fois terminés), vous pouvez lancer le serveur web local :
```bash
python -m http.server 8000
```
Puis ouvrez l'adresse suivante dans votre navigateur pour visualiser les résultats :
👉 **[http://localhost:8000/visualize_hgp.html](http://localhost:8000/visualize_hgp.html)**

---

## Fichiers clés :
* [visualize_hgp.html](file:///workspaces/Zoltan-SemanticKITTI/visualize_hgp.html) : Code du visualiseur 3D interactif.
* [src/generate_visualizations.py](file:///workspaces/Zoltan-SemanticKITTI/src/generate_visualizations.py) : Script de pré-calcul optimisé.
* `visualize/available_data.json` : Catalogue indiquant au navigateur quelles options charger (déjà pré-rempli dans le dépôt git par anticipation).

---

## 5. État actuel des calculs & Reprise (Mis à jour le 30 juin 2026)

Le script de génération a été amélioré pour supporter la reprise automatique en sautant les configurations déjà calculées.

### Avancement actuel :
* **Séquence 00** :
  * **Trame 0000** : **100% Complétée** (Géométrie + K=1..5 pour expZ=1.0..3.0).
  * **Trame 0001** : **Partiellement complétée** (Géométrie + K=1..4 complets. K=5 restant).
  * **Trames 0002 à 0004** : Non commencées.
* **Séquence 08** :
  * **Trames 0000 à 0004** : Non commencées.

### Optimisations et correctifs appliqués :
1. **Correction du cache Delaunay** : La clé de cache d'origine utilisait `id(M)`. Or, à chaque appel de `fit()`, du bruit aléatoire est ajouté aux coordonnées, générant une copie en mémoire avec un ID différent et rendant le cache inopérant. Nous l'avons remplacée par `(M.shape[0], K)`. Cela a activé le cache et apporté un **gain de vitesse de 5x** (la triangulation Delaunay d'ordre K n'est calculée qu'une seule fois par trame pour chaque valeur de K au lieu de se répéter pour chaque `expZ`).
2. **Mécanisme de Reprise (Skip)** : Le script vérifie désormais si les fichiers de clusters existent déjà et ne sont pas vides, sautant le calcul si c'est le cas.

### Pour reprendre le travail :
1. **Lancer la génération en tâche de fond** :
   ```bash
   python src/generate_visualizations.py
   ```
   *(Le script reprendra automatiquement à la Trame 0001, K=5)*

2. **Lancer le serveur de visualisation 3D** :
   ```bash
   python -m http.server 8000
   ```
   *(Accédez à http://localhost:8000/visualize_hgp.html)*


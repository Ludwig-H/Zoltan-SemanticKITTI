# Zoltan-SemanticKITTI

Ce projet explore un **renversement de paradigme** pour la segmentation panoptique (segmentation d'instance et suivi temporel) de nuages de points LiDAR 3D/4D sur le dataset **SemanticKITTI**.

## Le Paradigme : "Clustering-First, Semantics-Second"

Contrairement aux approches de l'état de l'art qui effectuent d'abord une segmentation sémantique globale puis regroupent les points sémantiquement similaires spatialement, ce projet propose l'approche inverse :

1. **Clustering Physique Fin (HGP-Clusterer)** : 
   * On segmente géométriquement la scène brute en clusters physiques cohérents à l'aide de l'algorithme topologique robuste [HGP-clusterer](file:///workspaces/Zoltan-SemanticKITTI/HGP-clusterer). 
   * Grâce à la percolation d'interactions d'ordre supérieur ($K \ge 2$), HGP isole efficacement les objets distincts même s'ils se touchent.
   * On configure l'algorithme pour favoriser une légère sur-segmentation contrôlée (plutôt que de risquer de fusionner des objets).
   
2. **Sémantique Localisée et Centrée** :
   * Les points au sein de chaque cluster candidat sont extraits et projetés dans un repère local centré sur leur propre centroïde $(0,0,0)$ et normalisés en échelle.
   * Un classifieur léger de points (type PointNet, Point Transformer ou GNN) est appliqué uniquement sur ces clusters isolés pour étiqueter sémantiquement chaque point (ex: différencier un piéton d'un cycliste dans un cluster fusionné).
   
3. **Découpage et Fusion Sémantico-Géométrique** :
   * Si un cluster contient des points classifiés dans des catégories sémantiques distinctes, il est scindé en plusieurs instances.
   * L'arbre hiérarchique (dendrogramme) fourni par HGP-Clusterer sert de guide pour trouver le bon niveau de coupure géométrique entre les objets.

---

## Jeu de Données SemanticKITTI

Le dataset SemanticKITTI nécessaire pour tester et entraîner nos modèles est disponible sur le Google Drive partagé :
👉 [SemanticKITTI Google Drive Folder](https://drive.google.com/drive/folders/1dvxpGeZuzvPJdSgFG2X2IWI2chu_kTZ9?usp=sharing)

---

## Structure du Dépôt

* [HGP-clusterer/](file:///workspaces/Zoltan-SemanticKITTI/HGP-clusterer) : Sous-module/librairie principale d'extraction géométrique.
* `src/` : Code source du pipeline (chargement des données, normalisation, modèles de classification locale, etc.).
* `data/` : Dossier destiné à accueillir le dataset localement (exclut du suivi Git).
* [.gitignore](file:///workspaces/Zoltan-SemanticKITTI/.gitignore) : Fichier de configuration des exclusions Git.

---

## Installation et Lancement

### Prerequis
* GCC/G++ >= 13.0
* CMake >= 3.15
* CGAL & TBB (pour la géométrie d'ordre supérieur)

Pour installer les bibliothèques requises sur Debian/Ubuntu :
```bash
sudo apt-get update && sudo apt-get install -y libcgal-dev libtbb-dev libeigen3-dev
```

### Compiler et Installer HGP-Clusterer
```bash
cd HGP-clusterer
pip install -e .
```
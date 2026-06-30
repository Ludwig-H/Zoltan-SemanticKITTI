import os
import glob
import numpy as np
from hgp_clusterer import HGPClusterer

class SemanticKITTILoader:
    """
    Charge les fichiers LiDAR (.bin) et sémantiques (.label) de SemanticKITTI.
    """
    def __init__(self, base_path, sequence_num):
        self.seq_str = f"{sequence_num:02d}"
        
        # Recherche du dossier de la séquence
        possible_paths = glob.glob(os.path.join(base_path, self.seq_str))
        if not possible_paths:
            possible_paths = glob.glob(os.path.join(base_path, "sequences", self.seq_str))
        if not possible_paths:
            possible_paths = glob.glob(os.path.join(base_path, "**", self.seq_str), recursive=True)

        valid_paths = [p for p in possible_paths if os.path.exists(os.path.join(p, 'velodyne'))]
        if not valid_paths:
            raise ValueError(f"Séquence {self.seq_str} introuvable dans {base_path}. Assurez-vous que 'velodyne' existe.")

        self.seq_path = valid_paths[0]
        print(f"[Loader] Séquence trouvée : {self.seq_path}")

        self.velo_path = os.path.join(self.seq_path, 'velodyne')
        self.label_path = os.path.join(self.seq_path, 'labels')
        
        self.scan_files = sorted(glob.glob(os.path.join(self.velo_path, '*.bin')))
        self.label_files = sorted(glob.glob(os.path.join(self.label_path, '*.label')))
        
    def load_frame(self, frame_idx):
        if frame_idx >= len(self.scan_files):
            raise IndexError(f"Index de frame {frame_idx} hors limite (total: {len(self.scan_files)})")
            
        # Chargement des points LiDAR (x, y, z, reflectance)
        points = np.fromfile(self.scan_files[frame_idx], dtype=np.float32).reshape(-1, 4)
        xyz = points[:, :3]
        
        # Chargement des labels sémantiques (si disponibles)
        labels = None
        if frame_idx < len(self.label_files):
            labels = np.fromfile(self.label_files[frame_idx], dtype=np.uint32)
            # Dans SemanticKITTI, l'ID d'instance est dans les 16 bits supérieurs
            # et l'ID sémantique est dans les 16 bits inférieurs
            semantic_labels = labels & 0xFFFF
            instance_labels = labels >> 16
        else:
            semantic_labels = np.zeros(xyz.shape[0], dtype=np.uint32)
            instance_labels = np.zeros(xyz.shape[0], dtype=np.uint32)
            
        return xyz, semantic_labels, instance_labels


def filter_ground_plane(xyz, height_threshold=-1.4):
    """
    Filtre géométrique simple pour exclure le sol (route, trottoir) sur l'axe Z.
    """
    non_ground_mask = xyz[:, 2] > height_threshold
    return xyz[non_ground_mask], non_ground_mask


def normalize_cluster(cluster_points):
    """
    Normalise localement les coordonnées des points d'un cluster :
    - Centre le cluster sur (0, 0, 0)
    - Normalise l'échelle pour que la distance max au centre soit 1.0.
    """
    centroid = np.mean(cluster_points, axis=0)
    centered = cluster_points - centroid
    
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    if max_dist > 1e-6:
        scaled = centered / max_dist
    else:
        scaled = centered
        
    return scaled, centroid, max_dist


class ParadigmShiftPipeline:
    """
    Pipeline implémentant le paradigme "Clustering-First, Semantics-Second".
    """
    def __init__(self, hgp_k=2, hgp_min_samples=5, hgp_min_cluster_size=15, backend='cgal'):
        self.clusterer = HGPClusterer(
            K=hgp_k,
            min_samples=hgp_min_samples,
            min_cluster_size=hgp_min_cluster_size,
            backend=backend,
            verbose=False
        )

    def process_frame(self, xyz_raw, sem_gt=None):
        print(f"\n--- Traitement d'une frame ({xyz_raw.shape[0]} points bruts) ---")
        
        # 1. Retrait du sol
        xyz_non_ground, non_ground_mask = filter_ground_plane(xyz_raw)
        print(f"[Étape 1] Points restants après filtrage du sol : {xyz_non_ground.shape[0]}")
        
        # 2. HGP-Clusterer (Clustering physique fin)
        print("[Étape 2] Lancement de HGP-Clusterer...")
        # Note: on utilise cgal/geogram optimisé
        labels = self.clusterer.fit_predict(xyz_non_ground)
        unique_labels = np.unique(labels)
        print(f"-> Clusters physiques détectés : {len(unique_labels[unique_labels >= 0])}")
        
        # 3. Normalisation locale et préparation pour le réseau
        print("[Étape 3] Normalisation locale des clusters...")
        prepared_clusters = {}
        for lab in unique_labels:
            if lab < 0:
                continue # Bruit géométrique exclu
                
            cluster_mask = (labels == lab)
            pts_cluster = xyz_non_ground[cluster_mask]
            
            # Normalisation locale
            norm_pts, centroid, max_dist = normalize_cluster(pts_cluster)
            
            # Récupérer la vérité terrain sémantique des points (si dispo pour l'évaluation)
            gt_sem_cluster = None
            if sem_gt is not None:
                gt_sem_cluster = sem_gt[non_ground_mask][cluster_mask]
                
            prepared_clusters[lab] = {
                'points_raw': pts_cluster,
                'points_normalized': norm_pts,
                'centroid': centroid,
                'max_dist': max_dist,
                'semantic_gt': gt_sem_cluster
            }
            
        print(f"-> {len(prepared_clusters)} clusters prêts pour la classification sémantique localisée.")
        return prepared_clusters, labels, non_ground_mask

    def run_mock_point_classification(self, prepared_clusters):
        """
        Simule la classification des points de chaque cluster par un réseau de neurones.
        En pratique, cette étape enverrait les points normalisés vers un réseau sémantique 3D local.
        """
        print("\n[Étape 4 (MOCK)] Simulation de classification sémantique locale...")
        for lab, data in prepared_clusters.items():
            norm_pts = data['points_normalized']
            n_pts = norm_pts.shape[0]
            
            # Simulation : Prédiction aléatoire de classes (ex: Piéton (class 1), Cycliste (class 2))
            # Dans un cas réel, PointNet(norm_pts) donnerait les prédictions
            mock_preds = np.random.choice([1, 2], size=n_pts, p=[0.7, 0.3])
            data['semantic_pred'] = mock_preds
            
            # Vérifier si on détecte une classe mixte au sein du même cluster physique
            unique_classes = np.unique(mock_preds)
            if len(unique_classes) > 1:
                print(f"  -> Cluster {lab} : Détection de plusieurs classes {unique_classes} ! "
                      f"Proposition de découpage en {len(unique_classes)} instances.")
                      
        return prepared_clusters


if __name__ == "__main__":
    # Test unitaire rapide du pipeline avec des données synthétiques
    print("=== Test Unitaire Local ===")
    
    # Génération de deux groupes de points (ex: un piéton et un cycliste proches l'un de l'autre)
    np.random.seed(42)
    pedestrian = np.random.normal(loc=[1.0, 0.0, 0.0], scale=0.1, size=(50, 3))
    cyclist = np.random.normal(loc=[1.3, 0.0, 0.0], scale=0.15, size=(80, 3))
    ground = np.random.uniform(low=[-5, -5, -1.6], high=[5, 5, -1.5], size=(200, 3))
    
    xyz_fake = np.vstack([ground, pedestrian, cyclist])
    
    pipeline = ParadigmShiftPipeline(hgp_k=2, hgp_min_samples=5, hgp_min_cluster_size=15, backend='cgal')
    
    # Exécution du pipeline
    prepared, labels, mask = pipeline.process_frame(xyz_fake)
    pipeline.run_mock_point_classification(prepared)

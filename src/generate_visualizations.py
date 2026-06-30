import os
import sys
import json
import numpy as np
import time

# Ajouter src au python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import SemanticKITTILoader, filter_ground_plane
from hgp_clusterer import HGPClusterer

# Monkeypatch orderk_delaunay3 pour mettre en cache les triangulations Delaunay d'ordre K.
# La triangulation de Delaunay d'ordre K ne dépend pas d'expZ, ce qui évite de recalculer
# la triangulation 3 fois par valeur de K sur chaque frame !
import hgp_clusterer.hypergraph

original_orderk_delaunay3 = hgp_clusterer.hypergraph.orderk_delaunay3
delaunay_cache = {}

def cached_orderk_delaunay3(M, K, *args, **kwargs):
    # La clé dépend de l'ID du tableau de coordonnées (constant par frame) et de K
    key = (id(M), K)
    if key not in delaunay_cache:
        delaunay_cache[key] = original_orderk_delaunay3(M, K, *args, **kwargs)
    return delaunay_cache[key]

hgp_clusterer.hypergraph.orderk_delaunay3 = cached_orderk_delaunay3


# Paramètres globaux de la sémantique
SEMANTIC_NAMES = {
    0: "unlabeled", 1: "outlier", 10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle",
    16: "on-rails", 18: "truck", 20: "other-vehicle", 30: "person", 31: "bicyclist",
    32: "motorcyclist", 40: "road", 44: "sidewalk", 48: "other-ground", 49: "building",
    50: "fence", 51: "other-structure", 70: "vegetation", 71: "trunk", 72: "terrain",
    80: "pole", 81: "traffic-sign"
}

SEMANTIC_COLORS = {
    0: '#cccccc', 1: '#999999', 10: '#004cff', 11: '#ffd400', 13: '#ff7c00', 15: '#ff0000',
    16: '#800000', 18: '#cc00ff', 20: '#5c007a', 30: '#ff1493', 31: '#ff007f', 32: '#8b008b',
    40: '#555555', 44: '#773377', 48: '#5c4033', 49: '#64a0f0', 50: '#006400', 51: '#a52a2a',
    70: '#00aa00', 71: '#8b5a2b', 72: '#7cfc00', 80: '#ff8800', 81: '#00ffff'
}

def generate_visualizations():
    data_dir = "/workspaces/Zoltan-SemanticKITTI/data"
    visualize_dir = "/workspaces/Zoltan-SemanticKITTI/visualize"
    os.makedirs(visualize_dir, exist_ok=True)
    
    # Configuration des séquences et frames à générer
    # Séquences disponibles : 00 (jusqu'à 16 frames) et 08 (jusqu'à 5 frames)
    configs = [
        {"seq": 0, "frames": list(range(5))},
        {"seq": 8, "frames": list(range(5))}
    ]
    
    Ks = [1, 2, 3, 4, 5]
    expZs = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    available_data = {
        "sequences": []
    }
    
    for conf in configs:
        seq_num = conf["seq"]
        frames = conf["frames"]
        seq_str = f"{seq_num:02d}"
        
        # Charger le loader pour cette séquence
        try:
            loader = SemanticKITTILoader(data_dir, sequence_num=seq_num)
        except Exception as e:
            print(f"Erreur d'initialisation pour la séquence {seq_str}: {e}")
            continue
            
        print(f"\n==========================================")
        print(f"Génération pour la Séquence {seq_str}...")
        print(f"==========================================")
        
        valid_frames = []
        
        for frame in frames:
            # Vider le cache de Delaunay à chaque nouvelle frame pour éviter la fuite de mémoire
            delaunay_cache.clear()
            
            if frame >= len(loader.scan_files):
                print(f"Frame {frame} non disponible pour la séquence {seq_str}. On arrête là.")
                break
                
            print(f"\n--- Traitement Sequence {seq_str} | Frame {frame} ---")
            xyz_raw, sem_gt, inst_gt = loader.load_frame(frame)
            xyz_non_ground, non_ground_mask = filter_ground_plane(xyz_raw)
            sem_gt_filtered = sem_gt[non_ground_mask]
            inst_gt_filtered = inst_gt[non_ground_mask]
            
            n_pts = xyz_non_ground.shape[0]
            print(f"Nombre de points après filtrage du sol : {n_pts}")
            
            if n_pts == 0:
                print("Pas de points dans cette frame. Ignorée.")
                continue
                
            # 1. Conversion des coordonnées SemanticKITTI -> Three.js
            # KITTI: X (avant), Y (gauche), Z (haut)
            # Three.js: X (droite), Y (haut), Z (arrière)
            positions_three = np.zeros_like(xyz_non_ground)
            positions_three[:, 0] = xyz_non_ground[:, 0]   # X -> X
            positions_three[:, 1] = xyz_non_ground[:, 2]   # Z -> Y
            positions_three[:, 2] = -xyz_non_ground[:, 1]  # -Y -> Z
            
            # Calcul des statistiques sémantiques réelles (GT) pour cette trame
            unique_classes = np.unique(sem_gt_filtered)
            class_stats = []
            for class_id in unique_classes:
                class_id = int(class_id)
                class_name = SEMANTIC_NAMES.get(class_id, f"Class {class_id}")
                class_mask = (sem_gt_filtered == class_id)
                pts_count = int(np.sum(class_mask))
                
                # Instance count (dans le nuage complet non-ground)
                inst_ids = inst_gt_filtered[class_mask]
                unique_insts = np.unique(inst_ids)
                unique_insts = unique_insts[unique_insts > 0]
                inst_count = len(unique_insts)
                
                color = SEMANTIC_COLORS.get(class_id, '#cccccc')
                class_stats.append({
                    'id': class_id,
                    'name': class_name,
                    'color': color,
                    'pts': pts_count,
                    'instances': inst_count
                })
            class_stats = sorted(class_stats, key=lambda x: x['pts'], reverse=True)
            
            # Sauvegarder le fichier de géométrie constant pour (seq, frame)
            geom_data = {
                "positions": positions_three.flatten().tolist(),
                "sem_gt": sem_gt_filtered.tolist(),
                "inst_gt": inst_gt_filtered.tolist(),
                "class_stats": class_stats
            }
            
            geom_file = f"data_seq{seq_str}_frame{frame:04d}.json"
            geom_filepath = os.path.join(visualize_dir, geom_file)
            print(f"Sauvegarde du fichier de géométrie : {geom_file}")
            with open(geom_filepath, "w") as f:
                json.dump(geom_data, f)
                
            # 2. Lancement du clustering pour chaque paramètre K, expZ
            for K in Ks:
                for expZ in expZs:
                    print(f"Lancement de HGP-Clusterer (K={K}, expZ={expZ:.1f}, complex_chosen='orderk_delaunay')...")
                    start_time = time.time()
                    
                    clusterer = HGPClusterer(
                        K=K,
                        expZ=expZ,
                        min_cluster_size=10,
                        complex_chosen='orderk_delaunay',
                        backend='geogram',
                        verbose=False
                    )
                    
                    labels = clusterer.fit_predict(xyz_non_ground)
                    elapsed = time.time() - start_time
                    
                    unique_labels = np.unique(labels)
                    num_clusters = int(np.sum(unique_labels >= 0))
                    
                    cluster_data = {
                        "labels": labels.tolist(),
                        "num_clusters": num_clusters,
                        "elapsed_seconds": elapsed
                    }
                    
                    cluster_file = f"clusters_seq{seq_str}_frame{frame:04d}_K{K}_expZ{expZ:.1f}.json"
                    cluster_filepath = os.path.join(visualize_dir, cluster_file)
                    with open(cluster_filepath, "w") as f:
                        json.dump(cluster_data, f)
                        
            valid_frames.append(frame)
            
        available_data["sequences"].append({
            "id": seq_str,
            "frames": valid_frames,
            "Ks": Ks,
            "expZs": expZs
        })
        
    # Sauvegarde du catalogue de données disponibles
    catalog_filepath = os.path.join(visualize_dir, "available_data.json")
    print(f"\nSauvegarde du catalogue de données : {catalog_filepath}")
    with open(catalog_filepath, "w") as f:
        json.dump(available_data, f, indent=2)
        
    print("Génération terminée avec succès !")

if __name__ == "__main__":
    generate_visualizations()

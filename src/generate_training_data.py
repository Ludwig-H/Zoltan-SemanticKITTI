import os
import sys
import numpy as np
from hgp_clusterer import HGPClusterer

# Ajouter src au python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import SemanticKITTILoader, filter_ground_plane, normalize_cluster

def generate_dataset(sequence_num=0, max_frames=10, K=2, min_samples=5, min_cluster_size=15):
    data_dir = "/workspaces/Zoltan-SemanticKITTI/data"
    output_dir = "/workspaces/Zoltan-SemanticKITTI/data/processed_clusters"
    os.makedirs(output_dir, exist_ok=True)
    
    loader = SemanticKITTILoader(data_dir, sequence_num=sequence_num)
    n_frames = min(max_frames, len(loader.scan_files))
    
    print(f"\n=== Génération du jeu de données d'entraînement (Séquence {sequence_num:02d}) ===")
    print(f"Traitement de {n_frames} frames...")
    
    clusterer = HGPClusterer(
        K=K,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        backend='cgal',
        verbose=False
    )
    
    total_clusters_saved = 0
    
    for frame_idx in range(n_frames):
        print(f"\nProcessing frame {frame_idx}/{n_frames}...")
        xyz_raw, sem_gt, inst_gt = loader.load_frame(frame_idx)
        
        # 1. Filtrer le sol
        xyz_non_ground, non_ground_mask = filter_ground_plane(xyz_raw)
        sem_gt_filtered = sem_gt[non_ground_mask]
        
        if xyz_non_ground.shape[0] < min_cluster_size:
            print(f"  -> Frame {frame_idx} ignorée (pas assez de points hors sol).")
            continue
            
        # 2. Exécuter HGP-Clusterer
        labels = clusterer.fit_predict(xyz_non_ground)
        unique_labels = np.unique(labels)
        
        # 3. Extraire et sauvegarder chaque cluster physique
        frame_clusters_count = 0
        for lab in unique_labels:
            if lab < 0:
                continue # Exclure le bruit
                
            cluster_mask = (labels == lab)
            pts_cluster = xyz_non_ground[cluster_mask]
            sem_cluster = sem_gt_filtered[cluster_mask]
            
            # Normaliser localement
            norm_pts, centroid, max_dist = normalize_cluster(pts_cluster)
            
            # Sauvegarder dans un fichier .npz compressé
            out_filename = f"seq{sequence_num:02d}_frame{frame_idx:04d}_cluster{lab:03d}.npz"
            out_path = os.path.join(output_dir, out_filename)
            
            np.savez_compressed(
                out_path,
                points_normalized=norm_pts.astype(np.float32),
                points_raw=pts_cluster.astype(np.float32),
                labels_semantic=sem_cluster.astype(np.int32),
                centroid=centroid.astype(np.float32),
                max_dist=np.float32(max_dist),
                sequence=np.int32(sequence_num),
                frame=np.int32(frame_idx),
                cluster_id=np.int32(lab)
            )
            frame_clusters_count += 1
            total_clusters_saved += 1
            
        print(f"  -> {frame_clusters_count} clusters extraits et sauvegardés dans {output_dir}.")
        
    print(f"\n=== Génération Terminée ===")
    print(f"Total de clusters d'entraînement générés : {total_clusters_saved}")

if __name__ == "__main__":
    # Générer le jeu de données pour les 5 premières frames de la séquence 00
    generate_dataset(sequence_num=0, max_frames=5)

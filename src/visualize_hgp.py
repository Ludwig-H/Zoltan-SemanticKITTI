import os
import sys
import argparse
import numpy as np
import plotly.graph_objects as go
from hgp_clusterer import HGPClusterer

# Ajouter src au path au cas où le script est exécuté depuis un sous-dossier
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import SemanticKITTILoader, filter_ground_plane

def main():
    parser = argparse.ArgumentParser(description="Visualisation interactive HGP-Clusterer sur SemanticKITTI")
    parser.add_argument("--sequence", type=int, default=8, help="Numéro de la séquence à utiliser")
    parser.add_argument("--frame", type=int, default=0, help="Index de la frame à visualiser")
    parser.add_argument("--K", type=int, default=2, help="Ordre des interactions de HGP-Clusterer (K=1, 2, ...)")
    parser.add_argument("--min_samples", type=int, default=5, help="Nombre de voisins pour l'estimateur de densité")
    parser.add_argument("--min_cluster_size", type=int, default=15, help="Taille minimale des clusters")
    parser.add_argument("--subsample", type=int, default=15, help="Facteur de sous-échantillonnage pour la fluidité 3D")
    parser.add_argument("--backend", type=str, default="cgal", choices=["cgal", "geogram"], help="Backend géométrique")
    parser.add_argument("--output", type=str, default="/workspaces/Zoltan-SemanticKITTI/visualize_hgp.html", help="Fichier HTML de sortie")
    
    # Gérer le cas Jupyter Notebook (où sys.argv contient des arguments Jupyter complexes)
    if 'ipykernel_launcher' in sys.argv[0] or '-f' in sys.argv:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
        
    # Auto-fallback si la séquence choisie n'existe pas localement ou est vide (pas de dossier velodyne)
    data_dir = "/workspaces/Zoltan-SemanticKITTI/data"
    seq_path_1 = os.path.join(data_dir, f"sequences/{args.sequence:02d}")
    seq_path_2 = os.path.join(data_dir, f"{args.sequence:02d}")
    has_velo = os.path.exists(os.path.join(seq_path_1, "velodyne")) or os.path.exists(os.path.join(seq_path_2, "velodyne"))
    
    if not has_velo:
        print(f"Séquence {args.sequence:02d} non trouvée localement ou incomplète. Recherche d'une séquence alternative...")
        seq_parent_dir = os.path.join(data_dir, "sequences")
        if os.path.exists(seq_parent_dir):
            available_seqs = sorted([
                d for d in os.listdir(seq_parent_dir) 
                if d.isdigit() and os.path.exists(os.path.join(seq_parent_dir, d, "velodyne"))
            ])
        else:
            available_seqs = []
            
        if available_seqs:
            args.sequence = int(available_seqs[0])
            print(f"Fallback automatique sur la Séquence {args.sequence:02d} !")
        else:
            raise FileNotFoundError(f"Aucune séquence complète avec dossier 'velodyne' trouvée dans {data_dir}")
    
    # 1. Chargement des données
    loader = SemanticKITTILoader(data_dir, sequence_num=args.sequence)
    
    print(f"Chargement de la frame {args.frame}...")
    xyz_raw, sem_gt, inst_gt = loader.load_frame(args.frame)
    
    # 2. Filtrage du sol
    xyz_non_ground, non_ground_mask = filter_ground_plane(xyz_raw)
    sem_gt_filtered = sem_gt[non_ground_mask]
    
    # 3. Lancement de HGP-Clusterer
    print(f"Lancement de HGP-Clusterer (K={args.K}, min_samples={args.min_samples}, min_cluster_size={args.min_cluster_size}, backend={args.backend})...")
    clusterer = HGPClusterer(
        K=args.K,
        min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size,
        backend=args.backend,
        verbose=True
    )
    labels = clusterer.fit_predict(xyz_non_ground)
    
    # 4. Sous-échantillonnage pour la fluidité d'affichage 3D dans le navigateur
    sub = args.subsample
    xyz_plot = xyz_non_ground[::sub]
    labels_plot = labels[::sub]
    sem_gt_plot = sem_gt_filtered[::sub]
    
    print(f"Points conservés pour l'affichage : {xyz_plot.shape[0]} (1 point sur {sub})")
    
    # 5. Création de la figure Plotly
    fig = go.Figure()
    
    # Séparer le bruit (label -1) des clusters
    noise_mask = (labels_plot == -1)
    cluster_mask = (labels_plot >= 0)
    
    # Tracer le bruit en gris clair transparent
    if np.any(noise_mask):
        fig.add_trace(go.Scatter3d(
            x=xyz_plot[noise_mask, 0],
            y=xyz_plot[noise_mask, 1],
            z=xyz_plot[noise_mask, 2],
            mode='markers',
            marker=dict(
                size=1.5,
                color='lightgrey',
                opacity=0.3
            ),
            name="Bruit (Label -1)",
            hoverinfo='text',
            text=[f"Point {i*sub} | GT Sem: {sem}" for i, sem in enumerate(sem_gt_plot[noise_mask])]
        ))
        
    # Tracer les clusters
    unique_labels = np.unique(labels_plot[cluster_mask])
    # Palette de couleurs distinctes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#e377c2']
    
    for idx, lab in enumerate(unique_labels):
        lab_mask = (labels_plot == lab)
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter3d(
            x=xyz_plot[lab_mask, 0],
            y=xyz_plot[lab_mask, 1],
            z=xyz_plot[lab_mask, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=color,
                opacity=0.9
            ),
            name=f"Cluster {lab}",
            hoverinfo='text',
            text=[f"Point {i*sub} | Cluster {lab} | GT Sem: {sem}" for i, sem in enumerate(sem_gt_plot[lab_mask])]
        ))
        
    fig.update_layout(
        title=f"HGP-Clusterer (Frame {args.frame}, K={args.K}, min_samples={args.min_samples}, min_cluster_size={args.min_cluster_size})",
        scene=dict(
            xaxis=dict(title='X (m)'),
            yaxis=dict(title='Y (m)'),
            zaxis=dict(title='Z (m)'),
            aspectmode='data'
        ),
        margin=dict(r=0, l=0, b=0, t=40)
    )
    
    fig.write_html(args.output)
    print(f"Visualisation sauvegardée dans : {args.output}")
    return fig

if __name__ == "__main__":
    main()

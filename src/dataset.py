import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class ClusterDataset(Dataset):
    """
    Dataset PyTorch chargeant les clusters physiques HGP normalisés
    pour entraîner un classifieur de points sémantiques.
    """
    def __init__(self, data_dir="/workspaces/Zoltan-SemanticKITTI/data/processed_clusters", transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
        if len(self.file_list) == 0:
            print(f"⚠️ Warning: Aucun fichier .npz trouvé dans {data_dir}. "
                  "Exécutez 'src/generate_training_data.py' d'abord.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path)
        
        # Charger les points normalisés (N, 3) et les labels sémantiques (N,)
        points = data['points_normalized'].astype(np.float32)
        labels = data['labels_semantic'].astype(np.int64)
        
        # Conversion en Tenseurs PyTorch
        points_tensor = torch.from_numpy(points)
        labels_tensor = torch.from_numpy(labels)
        
        # Application d'augmentations géométriques éventuelles
        if self.transform:
            points_tensor = self.transform(points_tensor)
            
        sample = {
            'points': points_tensor,
            'labels': labels_tensor,
            'centroid': torch.from_numpy(data['centroid']),
            'max_dist': torch.tensor(data['max_dist'], dtype=torch.float32),
            'filename': os.path.basename(file_path)
        }
        
        return sample

# ==============================================================================
# Outils d'Augmentation de Données 3D
# ==============================================================================

class RandomRotate3D:
    """ Rotation aléatoire autour de l'axe Z (très utile en LiDAR/conduite autonome) """
    def __call__(self, points):
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        # Matrice de rotation Z
        R = torch.tensor([
            [cos_t, -sin_t, 0],
            [sin_t,  cos_t, 0],
            [0,      0,     1]
        ], dtype=torch.float32)
        return torch.matmul(points, R.T)

class RandomJitter3D:
    """ Ajoute un léger bruit gaussien sur les coordonnées XYZ """
    def __init__(self, std=0.01, clip=0.05):
        self.std = std
        self.clip = clip

    def __call__(self, points):
        jitter = torch.randn_like(points) * self.std
        jitter = torch.clamp(jitter, -self.clip, self.clip)
        return points + jitter


if __name__ == "__main__":
    # Test unitaire rapide du dataset
    print("=== Test du Dataset PyTorch ===")
    from torch.utils.data import DataLoader
    
    # Instance avec augmentations
    dataset = ClusterDataset(
        transform=RandomJitter3D()
    )
    
    if len(dataset) > 0:
        print(f"Taille du jeu de données : {len(dataset)} clusters.")
        sample = dataset[0]
        print(f"Premier échantillon chargé : {sample['filename']}")
        print(f"  - Points (Tenseur)  : {sample['points'].shape}")
        print(f"  - Labels (Tenseur)  : {sample['labels'].shape}")
        print(f"  - Centroïde global : {sample['centroid']}")
        print(f"  - Rayon max         : {sample['max_dist']:.3f}m")
    else:
        print("Dataset vide.")

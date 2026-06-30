import os
import gdown

def download_dataset():
    # ID spécifique pour le dossier de la Séquence 08
    folder_id = "1UqFKvekjyic6L_8KD1kcv8MuGmQMIk0A"
    dest_dir = "/workspaces/Zoltan-SemanticKITTI/data/sequences/08"
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Analyse du dossier Google Drive de la Séquence 08 (ID: {folder_id})...")
    try:
        # Utilisation de gdown pour lister le contenu du dossier
        files = gdown.download_folder(id=folder_id, skip_download=True, quiet=True)
    except Exception as e:
        print(f"Erreur lors de la récupération de la liste des fichiers : {e}")
        return

    if not files:
        print("Aucun fichier trouvé. Veuillez vérifier les permissions de partage du dossier Google Drive.")
        return

    print(f"Nombre total de fichiers répertoriés dans la Séquence 08 : {len(files)}")
    
    download_count = 0
    skipped_count = 0
    
    for f in files:
        f_path = f.path if hasattr(f, 'path') else f.get('path', '')
        f_id = f.id if hasattr(f, 'id') else f.get('id', '')
        
        if not f_path or not f_id:
            continue
            
        # Nettoyer f_path si gdown a conservé le nom du dossier racine dans le chemin relatif
        # Par exemple, "08/velodyne/000000.bin" -> "velodyne/000000.bin"
        parts = f_path.split('/')
        if parts[0] in ["08", "sequences", "sequence"]:
            clean_path = "/".join(parts[1:])
        else:
            clean_path = f_path
            
        local_path = os.path.join(dest_dir, clean_path)
        filename = os.path.basename(local_path)
        parent_dir = os.path.basename(os.path.dirname(local_path))
        
        # Filtre : On ne télécharge que poses.txt, calib.txt, et les frames 000000 à 000049 (bin & label)
        should_download = False
        if filename in ["poses.txt", "calib.txt"]:
            should_download = True
        elif filename.endswith(".bin") and parent_dir == "velodyne":
            try:
                frame_idx = int(filename.split(".")[0])
                if frame_idx < 5:
                    should_download = True
            except ValueError:
                pass
        elif filename.endswith(".label") and parent_dir == "labels":
            try:
                frame_idx = int(filename.split(".")[0])
                if frame_idx < 5:
                    should_download = True
            except ValueError:
                pass
                
        if should_download:
            if not os.path.exists(local_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                print(f"Téléchargement : {clean_path} ...")
                try:
                    gdown.download(id=f_id, output=local_path, quiet=True)
                    download_count += 1
                except Exception as ex:
                    print(f"Erreur lors du téléchargement de {filename} : {ex}")
            else:
                skipped_count += 1
                
    print(f"\nTéléchargement terminé. {download_count} fichiers téléchargés, {skipped_count} déjà présents localement.")

if __name__ == "__main__":
    download_dataset()

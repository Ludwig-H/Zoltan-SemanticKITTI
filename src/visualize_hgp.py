import os
import sys
import json
import argparse
import numpy as np
from hgp_clusterer import HGPClusterer

# Ajouter src au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import SemanticKITTILoader, filter_ground_plane

def main():
    parser = argparse.ArgumentParser(description="Visualisation interactive 3D WebGL (Three.js) de HGP-Clusterer")
    parser.add_argument("--sequence", type=int, default=8, help="Numéro de la séquence à utiliser")
    parser.add_argument("--frame", type=int, default=0, help="Index de la frame à visualiser")
    parser.add_argument("--K", type=int, default=3, help="Ordre des interactions (K=1, 2, 3...)")
    parser.add_argument("--min_samples", type=int, default=None, help="Nombre de voisins pour l'estimateur de densité")
    parser.add_argument("--min_cluster_size", type=int, default=10, help="Taille minimale des clusters")
    parser.add_argument("--expZ", type=float, default=2.0, help="Exposant expZ")
    parser.add_argument("--subsample", type=int, default=1, help="Facteur de sous-échantillonnage (1 = tous les points)")
    parser.add_argument("--backend", type=str, default="geogram", choices=["cgal", "geogram"], help="Backend géométrique")
    parser.add_argument("--output", type=str, default="/workspaces/Zoltan-SemanticKITTI/visualize_hgp.html", help="Fichier HTML de sortie")
    
    if 'ipykernel_launcher' in sys.argv[0] or '-f' in sys.argv:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
        
    data_dir = "/workspaces/Zoltan-SemanticKITTI/data"
    
    # Auto-fallback si la séquence choisie n'existe pas localement ou est incomplète
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
        expZ=args.expZ,
        backend=args.backend,
        verbose=True
    )
    labels = clusterer.fit_predict(xyz_non_ground)
    
    # 4. Sous-échantillonnage pour la visualisation
    sub = args.subsample
    xyz_plot = xyz_non_ground[::sub]
    labels_plot = labels[::sub]
    sem_gt_plot = sem_gt_filtered[::sub]
    inst_gt_plot = inst_gt[non_ground_mask][::sub]
    
    print(f"Points conservés pour l'affichage : {xyz_plot.shape[0]} (1 point sur {sub})")
    
    # 5. Conversion des coordonnées SemanticKITTI -> Three.js
    # KITTI: X (avant), Y (gauche), Z (haut)
    # Three.js: X (droite), Y (haut), Z (arrière)
    positions_three = np.zeros_like(xyz_plot)
    positions_three[:, 0] = xyz_plot[:, 0]   # X -> X
    positions_three[:, 1] = xyz_plot[:, 2]   # Z -> Y
    positions_three[:, 2] = -xyz_plot[:, 1]  # -Y -> Z
    
    # Dictionnaires de correspondance pour la sémantique SemanticKITTI
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
    
    # Palette de couleurs distinctes hex pour HGP
    colors_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']
    
    # Conversion d'une couleur hex en float RGB [0, 1]
    def hex_to_rgb(hex_str):
        hex_str = hex_str.lstrip('#')
        return [int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
        
    rgb_palette = [hex_to_rgb(c) for c in colors_palette]
    rgb_sem_colors = {k: hex_to_rgb(v) for k, v in SEMANTIC_COLORS.items()}
    
    # 6. Génération des Buffers GPU en Python
    n_pts = xyz_plot.shape[0]
    
    # Buffers HGP
    hgp_rgb = np.zeros((n_pts, 3), dtype=np.float32)
    hgp_sizes = np.zeros(n_pts, dtype=np.float32)
    hgp_opacities = np.zeros(n_pts, dtype=np.float32)
    
    unique_labels = np.unique(labels_plot)
    for idx, lab in enumerate(unique_labels):
        mask = (labels_plot == lab)
        if lab == -1:
            hgp_rgb[mask] = [0.8, 0.8, 0.8]  # gris clair pour le bruit
            hgp_sizes[mask] = 1.0
            hgp_opacities[mask] = 0.15
        else:
            hgp_rgb[mask] = rgb_palette[idx % len(rgb_palette)]
            hgp_sizes[mask] = 2.0
            hgp_opacities[mask] = 0.65
            
    # Buffers Sémantique
    sem_rgb = np.zeros((n_pts, 3), dtype=np.float32)
    sem_sizes = np.zeros(n_pts, dtype=np.float32)
    sem_opacities = np.zeros(n_pts, dtype=np.float32)
    
    for class_id in np.unique(sem_gt_plot):
        mask = (sem_gt_plot == class_id)
        color = rgb_sem_colors.get(class_id, [0.8, 0.8, 0.8])
        is_ground = class_id in [40, 44, 48, 72]
        
        sem_rgb[mask] = color
        sem_sizes[mask] = 1.5 if is_ground else 2.5
        sem_opacities[mask] = 0.20 if is_ground else 0.80
        
    # Buffers Instances GT
    inst_rgb = np.zeros((n_pts, 3), dtype=np.float32)
    inst_sizes = np.zeros(n_pts, dtype=np.float32)
    inst_opacities = np.zeros(n_pts, dtype=np.float32)
    
    unique_instances_plot = np.unique(inst_gt_plot)
    for idx, inst_id in enumerate(unique_instances_plot):
        mask = (inst_gt_plot == inst_id)
        if inst_id == 0:
            inst_rgb[mask] = [0.8, 0.8, 0.8]  # gris clair pour le non-instancié (sol, arbres...)
            inst_sizes[mask] = 1.0
            inst_opacities[mask] = 0.15
        else:
            inst_rgb[mask] = rgb_palette[idx % len(rgb_palette)]
            inst_sizes[mask] = 2.0
            inst_opacities[mask] = 0.80
        
    # --- CALCUL DES STATISTIQUES POUR LE DASHBOARD ---
    total_points_raw = xyz_non_ground.shape[0]
    num_clusters = len(unique_labels[unique_labels >= 0])
    
    unique_classes_all = np.unique(sem_gt_filtered)
    class_stats = []
    for class_id in unique_classes_all:
        class_name = SEMANTIC_NAMES.get(class_id, f"Class {class_id}")
        class_mask = (sem_gt_filtered == class_id)
        pts_count = np.sum(class_mask)
        
        # Instance count (dans le nuage complet non-ground)
        inst_ids = inst_gt[non_ground_mask][class_mask]
        unique_insts = np.unique(inst_ids)
        unique_insts = unique_insts[unique_insts > 0]
        inst_count = len(unique_insts)
        
        color = SEMANTIC_COLORS.get(class_id, '#cccccc')
        class_stats.append({
            'name': class_name,
            'color': color,
            'pts': pts_count,
            'instances': inst_count
        })
    class_stats = sorted(class_stats, key=lambda x: x['pts'], reverse=True)
    
    # Génération du code HTML pour les lignes de stats
    table_rows_html = ""
    for stat in class_stats:
        inst_display = stat['instances'] if stat['instances'] > 0 else "-"
        table_rows_html += f"""
        <tr>
            <td>
                <div class="class-name-col">
                    <span class="color-dot" style="background-color: {stat['color']};"></span>
                    {stat['name']}
                </div>
            </td>
            <td class="align-right">{stat['pts']:,}</td>
            <td class="align-right">{inst_display}</td>
        </tr>
        """
        
    # 7. Écriture du fichier HTML WebGL complet
    html_template = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HGP-Clusterer 3D WebGL Viewer</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ margin: 0; padding: 0; overflow: hidden; background-color: #0c0c0e; font-family: 'Inter', sans-serif; color: #f3f4f6; }}
        #canvas-container {{ width: 100vw; height: 100vh; position: absolute; z-index: 1; }}
        
        /* Dashboard de statistiques */
        #stats-dashboard {{
            position: absolute;
            top: 20px;
            right: 20px;
            width: 330px;
            max-height: 90vh;
            overflow-y: auto;
            background: rgba(18, 18, 18, 0.75);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 20px;
            z-index: 10;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }}
        #stats-dashboard::-webkit-scrollbar {{
            width: 6px;
        }}
        #stats-dashboard::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.15);
            border-radius: 3px;
        }}
        .dashboard-title {{
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 15px;
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 10px;
        }}
        .section-title {{
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #9ca3af;
            margin: 15px 0 8px 0;
            border-bottom: 1px dashed rgba(255, 255, 255, 0.05);
            padding-bottom: 3px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            margin-bottom: 6px;
        }}
        .stat-label {{
            color: #9ca3af;
        }}
        .stat-value {{
            font-weight: 600;
            color: #ffffff;
        }}
        .class-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 5px;
            font-size: 12px;
        }}
        .class-table th {{
            text-align: left;
            color: #9ca3af;
            font-weight: 500;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .class-table td {{
            padding: 5px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }}
        .class-name-col {{
            display: flex;
            align-items: center;
        }}
        .color-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
            display: inline-block;
        }}
        .align-right {{
            text-align: right;
        }}
        
        /* Boutons de contrôle de vue */
        #control-panel {{
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 10;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .btn {{
            background: rgba(18, 18, 18, 0.75);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #fff;
            padding: 10px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            font-size: 13px;
            transition: all 0.2s ease;
            height: 38px;
        }}
        .btn:hover {{
            background: rgba(255, 255, 255, 0.15);
            border-color: rgba(255, 255, 255, 0.3);
        }}
        .btn.active {{
            background: #3b82f6;
            border-color: #60a5fa;
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
        }}
        
        /* Slider de taille des points */
        .slider-container {{
            display: flex;
            align-items: center;
            background: rgba(18, 18, 18, 0.75);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 0 15px;
            color: #fff;
            font-size: 13px;
            font-weight: 600;
            height: 38px;
        }}
        .slider-container label {{
            margin-right: 10px;
            color: #9ca3af;
        }}
        input[type="range"] {{
            cursor: pointer;
            accent-color: #3b82f6;
            width: 100px;
        }}
        #size-val {{
            margin-left: 10px;
            min-width: 25px;
            text-align: right;
            color: #60a5fa;
        }}
    </style>
</head>
<body>
    <div id="control-panel">
        <button id="btn-hgp" class="btn active" onclick="setViewMode('hgp')">Visualiser Clusters HGP</button>
        <button id="btn-sem" class="btn" onclick="setViewMode('sem')">Visualiser Sémantique (GT)</button>
        <button id="btn-inst" class="btn" onclick="setViewMode('inst')">Visualiser Instances (GT)</button>
        
        <div class="slider-container">
            <label for="size-slider">Taille Points :</label>
            <input type="range" id="size-slider" min="0.1" max="3.0" step="0.1" value="0.6" oninput="updatePointSize(this.value)">
            <span id="size-val">0.6</span>
        </div>
    </div>
    
    <div id="stats-dashboard">
        <div class="dashboard-title">HGP-Clusterer Dashboard</div>
        
        <div class="section-title">Informations de Frame</div>
        <div class="stat-row"><span class="stat-label">Séquence</span><span class="stat-value">{args.sequence:02d}</span></div>
        <div class="stat-row"><span class="stat-label">Index Frame</span><span class="stat-value">{args.frame}</span></div>
        <div class="stat-row"><span class="stat-label">Backend</span><span class="stat-value">{args.backend.upper()}</span></div>
        
        <div class="section-title">Paramètres HGP</div>
        <div class="stat-row"><span class="stat-label">Ordre K</span><span class="stat-value">{args.K}</span></div>
        <div class="stat-row"><span class="stat-label">min_samples</span><span class="stat-value">{args.min_samples if args.min_samples is not None else f"None (Auto={args.K+1})"}</span></div>
        <div class="stat-row"><span class="stat-label">min_cluster_size</span><span class="stat-value">{args.min_cluster_size}</span></div>
        <div class="stat-row"><span class="stat-label">expZ</span><span class="stat-value">{args.expZ}</span></div>
        
        <div class="section-title">Statistiques Globales</div>
        <div class="stat-row"><span class="stat-label">Points Totaux (Hors-sol)</span><span class="stat-value">{total_points_raw:,}</span></div>
        <div class="stat-row"><span class="stat-label">Facteur Sous-échantillon.</span><span class="stat-value">1 / {sub}</span></div>
        <div class="stat-row"><span class="stat-label">Points Affichés</span><span class="stat-value">{n_pts:,}</span></div>
        <div class="stat-row"><span class="stat-label">Clusters HGP Détectés</span><span class="stat-value">{num_clusters}</span></div>
        
        <div class="section-title">Classes Sémantiques (GT)</div>
        <table class="class-table">
            <thead>
                <tr>
                    <th>Classe</th>
                    <th class="align-right">Points</th>
                    <th class="align-right">Instances</th>
                </tr>
            </thead>
            <tbody>
                {table_rows_html}
            </tbody>
        </table>
    </div>

    <div id="canvas-container"></div>

    <!-- Scripts CDN pour Three.js -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

    <script>
        // --- 1. CHARGEMENT DES BUFFERS DÉFINIS PAR PYTHON ---
        const pointsData = {{
            positions: new Float32Array({json.dumps(positions_three.flatten().tolist())}),
            
            hgpColors: new Float32Array({json.dumps(hgp_rgb.flatten().tolist())}),
            hgpSizes: new Float32Array({json.dumps(hgp_sizes.tolist())}),
            hgpOpacities: new Float32Array({json.dumps(hgp_opacities.tolist())}),
            
            semColors: new Float32Array({json.dumps(sem_rgb.flatten().tolist())}),
            semSizes: new Float32Array({json.dumps(sem_sizes.tolist())}),
            semOpacities: new Float32Array({json.dumps(sem_opacities.tolist())}),
            
            instColors: new Float32Array({json.dumps(inst_rgb.flatten().tolist())}),
            instSizes: new Float32Array({json.dumps(inst_sizes.tolist())}),
            instOpacities: new Float32Array({json.dumps(inst_opacities.tolist())})
        }};

        // --- 2. CONFIGURATION DE LA SCÈNE THREE.JS ---
        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x0c0c0e, 0.005);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 40, 80);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.maxPolarAngle = Math.PI / 2 + 0.1; // limite pour ne pas passer sous la grille

        // Ajouter une grille de sol
        const gridHelper = new THREE.GridHelper(300, 150, 0x444444, 0x1a1a1f);
        gridHelper.position.y = -2;
        scene.add(gridHelper);

        // --- 3. CUSTOM SHADER MATERIAL POUR LES POINTS CLOUDS ---
        // Permet de dessiner des points parfaitement ronds avec gestion de l'opacité et de la taille individuelle
        const vertexShader = `
            uniform float sizeMultiplier;
            attribute float size;
            attribute float opacity;
            varying vec3 vColor;
            varying float vOpacity;
            void main() {{
                vColor = color;
                vOpacity = opacity;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                // Perspective de taille pondérée par le multiplier interactif
                gl_PointSize = size * sizeMultiplier * (300.0 / -mvPosition.z);
                gl_Position = projectionMatrix * mvPosition;
            }}
        `;

        const fragmentShader = `
            varying vec3 vColor;
            varying float vOpacity;
            void main() {{
                // Transforme le carré de point gl_PointCoord en rond
                vec2 temp = gl_PointCoord - vec2(0.5);
                if (dot(temp, temp) > 0.25) discard;
                gl_FragColor = vec4(vColor, vOpacity);
            }}
        `;

        // --- 4. CRÉATION DU MESH DE POINTS CLOUD ---
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(pointsData.positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(pointsData.hgpColors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(pointsData.hgpSizes, 1));
        geometry.setAttribute('opacity', new THREE.BufferAttribute(pointsData.hgpOpacities, 1));

        const material = new THREE.ShaderMaterial({{
            uniforms: {{
                sizeMultiplier: {{ value: 0.6 }} // Définit la taille initiale plus petite (0.6 au lieu de 1.0)
            }},
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            transparent: true,
            depthWrite: false, // évite les artefacts de superposition des points transparents
            vertexColors: true
        }});

        const pointCloud = new THREE.Points(geometry, material);
        scene.add(pointCloud);

        // Centrer la caméra sur le centre de gravité des points
        let sumX = 0, sumY = 0, sumZ = 0;
        const count = pointsData.positions.length / 3;
        for (let i = 0; i < pointsData.positions.length; i += 3) {{
            sumX += pointsData.positions[i];
            sumY += pointsData.positions[i+1];
            sumZ += pointsData.positions[i+2];
        }}
        const centroid = new THREE.Vector3(sumX / count, sumY / count, sumZ / count);
        controls.target.copy(centroid);
        camera.position.set(centroid.x, centroid.y + 30, centroid.z + 60);
        controls.update();

        // --- 5. INTERACTIVITÉ (SWITCH MODE & SLIDER) ---
        function setViewMode(mode) {{
            const btnHGP = document.getElementById('btn-hgp');
            const btnSem = document.getElementById('btn-sem');
            const btnInst = document.getElementById('btn-inst');
            
            btnHGP.classList.remove('active');
            btnSem.classList.remove('active');
            btnInst.classList.remove('active');
            
            if (mode === 'hgp') {{
                btnHGP.classList.add('active');
                geometry.setAttribute('color', new THREE.BufferAttribute(pointsData.hgpColors, 3));
                geometry.setAttribute('size', new THREE.BufferAttribute(pointsData.hgpSizes, 1));
                geometry.setAttribute('opacity', new THREE.BufferAttribute(pointsData.hgpOpacities, 1));
            }} else if (mode === 'sem') {{
                btnSem.classList.add('active');
                geometry.setAttribute('color', new THREE.BufferAttribute(pointsData.semColors, 3));
                geometry.setAttribute('size', new THREE.BufferAttribute(pointsData.semSizes, 1));
                geometry.setAttribute('opacity', new THREE.BufferAttribute(pointsData.semOpacities, 1));
            }} else if (mode === 'inst') {{
                btnInst.classList.add('active');
                geometry.setAttribute('color', new THREE.BufferAttribute(pointsData.instColors, 3));
                geometry.setAttribute('size', new THREE.BufferAttribute(pointsData.instSizes, 1));
                geometry.setAttribute('opacity', new THREE.BufferAttribute(pointsData.instOpacities, 1));
            }}
            
            geometry.attributes.color.needsUpdate = true;
            geometry.attributes.size.needsUpdate = true;
            geometry.attributes.opacity.needsUpdate = true;
        }}

        function updatePointSize(val) {{
            material.uniforms.sizeMultiplier.value = parseFloat(val);
            document.getElementById('size-val').innerText = val;
        }}

        // --- 6. RENDER LOOP & RESPONSIVITÉ ---
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
"""
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_template)
        
    print(f"Visualisation 3D WebGL (Three.js) sauvegardée dans : {args.output}")

if __name__ == "__main__":
    main()

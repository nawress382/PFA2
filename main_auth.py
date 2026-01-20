"""
Système d'Authentification Faciale par Pose
============================================
Utilise dlib pour la détection faciale et l'estimation de pose 3D
pour une authentification sécurisée multi-facteurs.

Auteur: Optimisé pour une utilisation professionnelle
"""

import cv2
import dlib
import numpy as np
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration centralisée du système"""
    SHAPE_PREDICTOR_PATH: str = "shape_predictor_68_face_landmarks.dat"
    FACE_RECOGNITION_MODEL_PATH: str = "dlib_face_recognition_resnet_model_v1.dat"
    DATASET_DIR: str = "dataset"
    
    # Seuils et paramètres
    FACE_DETECTION_THRESHOLD: float = 0.6
    TOLERANCE_YAW: float = 15.0
    TOLERANCE_PITCH: float = 15.0
    TOLERANCE_ROLL: float = 15.0
    POSE_DURATION: float = 3.0
    CAPTURE_DELAY: float = 0.5
    
    # Paramètres caméra
    FRAME_SCALE: float = 0.75
    UPSAMPLE_TIMES: int = 1
    
    # Poses disponibles
    POSES: List[str] = None
    
    def __post_init__(self):
        if self.POSES is None:
            self.POSES = ["frontale", "gauche", "droite", "haut", "bas"]


# ============================================================================
# CLASSE PRINCIPALE - SYSTÈME DE RECONNAISSANCE
# ============================================================================

class FacialRecognitionSystem:
    """Système complet de reconnaissance faciale avec authentification par pose"""
    
    def __init__(self, config: Config):
        self.config = config
        self._initialize_models()
        self._setup_camera_parameters()
        
    def _initialize_models(self):
        """Initialisation des modèles dlib avec validation"""
        logger.info("Initialisation des modèles dlib...")
        
        # Détecteur de visages
        self.detector = dlib.get_frontal_face_detector()
        
        # Prédicteur de landmarks
        if not Path(self.config.SHAPE_PREDICTOR_PATH).exists():
            raise FileNotFoundError(
                f"Modèle shape_predictor introuvable: {self.config.SHAPE_PREDICTOR_PATH}\n"
                "Téléchargez shape_predictor_68_face_landmarks.dat.bz2"
            )
        self.predictor = dlib.shape_predictor(self.config.SHAPE_PREDICTOR_PATH)
        
        # Modèle de reconnaissance
        if not Path(self.config.FACE_RECOGNITION_MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Modèle de reconnaissance introuvable: {self.config.FACE_RECOGNITION_MODEL_PATH}\n"
                "Téléchargez dlib_face_recognition_resnet_model_v1.dat.bz2"
            )
        self.face_recognizer = dlib.face_recognition_model_v1(
            self.config.FACE_RECOGNITION_MODEL_PATH
        )
        
        logger.info("✓ Modèles initialisés avec succès")
    
    def _setup_camera_parameters(self):
        """Configuration des paramètres de la caméra pour l'estimation de pose"""
        self.camera_matrix = np.array([
            [0.0, 0, 0],
            [0, 0.0, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.dist_coeffs = np.zeros((4, 1))
        
        # Points du modèle 3D du visage
        self.model_points = np.array([
            (0.0, 0.0, 0.0),           # Nez (30)
            (0.0, -330.0, -65.0),      # Menton (8)
            (-225.0, 170.0, -135.0),   # Coin œil gauche (36)
            (225.0, 170.0, -135.0),    # Coin œil droit (45)
            (-150.0, -150.0, -125.0),  # Coin bouche gauche (48)
            (150.0, -150.0, -125.0)    # Coin bouche droit (54)
        ], dtype=np.float64)
    
    def get_face_encoding(self, image_rgb: np.ndarray, face_rect: dlib.rectangle) -> np.ndarray:
        """Calcule l'encodage facial pour un visage détecté"""
        landmarks = self.predictor(image_rgb, face_rect)
        return np.array(self.face_recognizer.compute_face_descriptor(image_rgb, landmarks))
    
    def estimate_pose(self, landmarks: dlib.full_object_detection, 
                     img_w: int, img_h: int) -> Tuple[float, float, float]:
        """Estime la pose 3D du visage (yaw, pitch, roll)"""
        # Mise à jour de la matrice caméra
        focal_length = img_w
        self.camera_matrix[0, 0] = focal_length
        self.camera_matrix[1, 1] = focal_length
        self.camera_matrix[0, 2] = img_w / 2
        self.camera_matrix[1, 2] = img_h / 2
        
        # Points d'intérêt sur le visage
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),
            (landmarks.part(8).x, landmarks.part(8).y),
            (landmarks.part(36).x, landmarks.part(36).y),
            (landmarks.part(45).x, landmarks.part(45).y),
            (landmarks.part(48).x, landmarks.part(48).y),
            (landmarks.part(54).x, landmarks.part(54).y)
        ], dtype=np.float64)
        
        # Résolution PnP
        success, rvec, tvec = cv2.solvePnP(
            self.model_points, image_points, 
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0
        
        # Conversion en angles d'Euler
        rot_matrix, _ = cv2.Rodrigues(rvec)
        proj_matrix = np.hstack((rot_matrix, tvec))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        
        pitch, yaw, roll = euler_angles[0, 0], euler_angles[1, 0], euler_angles[2, 0]
        
        return -yaw, pitch, -roll
    
    def capture_dataset(self, user_name: str, num_images_per_pose: int = 10):
        """Capture d'images pour créer un dataset utilisateur"""
        output_dir = Path(self.config.DATASET_DIR) / user_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Impossible d'ouvrir la caméra")
            return
        
        logger.info(f"📸 Capture pour {user_name} - {len(self.config.POSES)} poses")
        logger.info("Appuyez sur 's' pour capturer, 'q' pour quitter")
        
        for pose_name in self.config.POSES:
            logger.info(f"\n🎯 Pose: {pose_name.upper()} ({num_images_per_pose} images)")
            time.sleep(2)
            
            count = 0
            while count < num_images_per_pose:
                ret, frame = cap.read()
                if not ret:
                    break
                
                display = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                
                # Interface utilisateur
                self._draw_capture_ui(display, faces, pose_name, count, num_images_per_pose)
                cv2.imshow('Capture Dataset', display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s') and len(faces) == 1:
                    img_path = output_dir / f"{user_name}_{pose_name}_{count:03d}.jpg"
                    cv2.imwrite(str(img_path), frame)
                    logger.info(f"✓ Image {count + 1}/{num_images_per_pose} sauvegardée")
                    count += 1
                    time.sleep(self.config.CAPTURE_DELAY)
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"✓ Capture terminée pour {user_name}")
    
    def _draw_capture_ui(self, frame: np.ndarray, faces: List, 
                        pose_name: str, count: int, total: int):
        """Dessine l'interface de capture avec design moderne"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Barre supérieure avec gradient
        cv2.rectangle(overlay, (0, 0), (w, 100), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        
        if len(faces) == 1:
            face = faces[0]
            x, y, width, height = face.left(), face.top(), face.width(), face.height()
            
            # Rectangle de visage avec coins arrondis style moderne
            self._draw_modern_face_box(frame, x, y, width, height, (0, 255, 180), 3)
            
            # Indicateur de progression avec barre
            progress = (count / total) * 100
            cv2.putText(frame, f"POSE: {pose_name.upper()}", 
                       (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(frame, f"{count + 1}/{total}", 
                       (w - 120, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 180), 2)
            
            # Barre de progression
            bar_y = 75
            cv2.rectangle(frame, (20, bar_y), (w - 20, bar_y + 10), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, bar_y), (int(20 + (w - 40) * progress / 100), bar_y + 10), (0, 255, 180), -1)
            
            # Instructions en bas avec fond semi-transparent
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (0, h - 80), (w, h), (20, 20, 20), -1)
            frame = cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0)
            cv2.putText(frame, "Appuyez sur [S] pour capturer", 
                       (w // 2 - 200, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # Message d'alerte centré
            cv2.putText(frame, "POSITIONNEZ VOTRE VISAGE", 
                       (w // 2 - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)
            # Cercle guide au centre
            center_x, center_y = w // 2, h // 2
            cv2.circle(frame, (center_x, center_y), 120, (0, 100, 255), 3)
            cv2.circle(frame, (center_x, center_y), 122, (255, 255, 255), 1)
    
    def load_dataset(self) -> Tuple[np.ndarray, List[str], Dict]:
        """Charge et traite le dataset d'encodages faciaux"""
        dataset_path = Path(self.config.DATASET_DIR)
        if not dataset_path.exists():
            logger.warning(f"Dataset introuvable: {dataset_path}")
            return np.array([]), [], {}
        
        known_encodings = []
        known_names = []
        known_poses = {}
        
        logger.info("📂 Chargement du dataset...")
        
        for user_dir in dataset_path.iterdir():
            if not user_dir.is_dir():
                continue
            
            user_name = user_dir.name
            logger.info(f"  → Traitement: {user_name}")
            
            user_encodings = []
            user_poses_data = {pose: [] for pose in self.config.POSES}
            
            for img_path in user_dir.glob("*.jpg"):
                try:
                    encoding, pose_angles = self._process_image(img_path)
                    if encoding is not None:
                        user_encodings.append(encoding)
                        
                        # Association de la pose
                        for pose_name in self.config.POSES:
                            if pose_name in img_path.stem.lower():
                                user_poses_data[pose_name].append(pose_angles)
                                break
                except Exception as e:
                    logger.error(f"Erreur sur {img_path.name}: {e}")
            
            # Moyennes des encodages et poses
            if user_encodings:
                known_encodings.append(np.mean(user_encodings, axis=0))
                known_names.append(user_name)
                
                user_avg_poses = {}
                for pose_name, angles_list in user_poses_data.items():
                    if angles_list:
                        user_avg_poses[pose_name] = np.mean(angles_list, axis=0)
                
                known_poses[user_name] = user_avg_poses
                logger.info(f"  ✓ {len(user_encodings)} images, {len(user_avg_poses)} poses")
        
        logger.info(f"✓ Dataset chargé: {len(known_names)} utilisateurs")
        return np.array(known_encodings), known_names, known_poses
    
    def _process_image(self, img_path: Path) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
        """Traite une image pour extraire l'encodage et la pose"""
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None, None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector(gray, self.config.UPSAMPLE_TIMES)
        
        if len(faces) != 1:
            return None, None
        
        face = faces[0]
        encoding = self.get_face_encoding(img_rgb, face)
        landmarks = self.predictor(img_rgb, face)
        h, w = img_rgb.shape[:2]
        pose_angles = self.estimate_pose(landmarks, w, h)
        
        return encoding, pose_angles
    
    def authenticate(self, known_encodings: np.ndarray, 
                    known_names: List[str], known_poses: Dict):
        """Lance le processus d'authentification par reconnaissance faciale et pose"""
        if len(known_encodings) == 0:
            logger.error("Aucun utilisateur enregistré. Capturez d'abord un dataset.")
            return
        
        # Initialisation du KNN
        knn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        knn.fit(known_encodings)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Impossible d'ouvrir la caméra")
            return
        
        # État d'authentification
        state = AuthenticationState()
        
        logger.info("🔐 Authentification démarrée - Appuyez sur 'Q' pour quitter")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Traitement de la frame
            processed = self._process_authentication_frame(
                frame, knn, known_encodings, known_names, known_poses, state
            )
            
            cv2.imshow('Authentification Faciale', processed)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Authentification terminée")
    
    def _process_authentication_frame(self, frame: np.ndarray, knn, 
                                      known_encodings, known_names, 
                                      known_poses, state) -> np.ndarray:
        """Traite une frame pour l'authentification"""
        h, w = frame.shape[:2]
        display = frame.copy()
        
        # Détection sur frame réduite pour performance
        small = cv2.resize(frame, (0, 0), 
                          fx=self.config.FRAME_SCALE, 
                          fy=self.config.FRAME_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector(gray_small, self.config.UPSAMPLE_TIMES)
        
        if len(faces) == 0:
            self._draw_no_face(display)
            state.reset()
            return display
        
        # Traitement du premier visage
        face = faces[0]
        face_scaled = self._scale_face_rect(face, 1 / self.config.FRAME_SCALE)
        
        # Landmarks et pose sur frame originale
        landmarks = self.predictor(frame, face_scaled)
        yaw, pitch, roll = self.estimate_pose(landmarks, w, h)
        
        # Dessin du rectangle
        self._draw_face_rect(display, face_scaled)
        self._draw_pose_overlay(display, yaw, pitch, roll)
        
        # Logique d'authentification
        if state.stage == "IDLE":
            self._handle_identification(
                rgb_small, face, knn, known_names, known_poses, state, display
            )
        elif state.stage == "POSE_CHALLENGE":
            self._handle_pose_challenge(
                yaw, pitch, roll, known_poses, state, display, w, h
            )
        
        return display
    
    def _scale_face_rect(self, rect: dlib.rectangle, scale: float) -> dlib.rectangle:
        """Redimensionne un rectangle de visage"""
        return dlib.rectangle(
            int(rect.left() * scale), int(rect.top() * scale),
            int(rect.right() * scale), int(rect.bottom() * scale)
        )
    
    def _handle_identification(self, rgb_frame, face, knn, known_names, 
                              known_poses, state, display):
        """Gère la phase d'identification"""
        try:
            encoding = self.get_face_encoding(rgb_frame, face)
            distances, indices = knn.kneighbors([encoding])
            
            if distances[0][0] < self.config.FACE_DETECTION_THRESHOLD:
                user = known_names[indices[0][0]]
                cv2.putText(display, f"✓ Identifié: {user}", 
                           (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
                
                if user in known_poses and known_poses[user]:
                    state.set_user(user, list(known_poses[user].keys()))
                    logger.info(f"🎯 Défi de pose pour {user}: {state.required_pose}")
                else:
                    cv2.putText(display, "⚠ Pas de poses enregistrées", 
                               (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                cv2.putText(display, "✗ Inconnu", 
                           (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
        except Exception as e:
            logger.error(f"Erreur identification: {e}")
    
    def _handle_pose_challenge(self, yaw, pitch, roll, known_poses, 
                              state, display, w, h):
        """Gère le défi de pose"""
        cv2.putText(display, f"Utilisateur: {state.authenticated_user}", 
                   (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(display, f"🎯 Pose requise: {state.required_pose.upper()}", 
                   (10, 180), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
        
        target = known_poses[state.authenticated_user][state.required_pose]
        
        # Vérification de la correspondance
        yaw_ok = abs(yaw - target[0]) < self.config.TOLERANCE_YAW
        pitch_ok = abs(pitch - target[1]) < self.config.TOLERANCE_PITCH
        roll_ok = abs(roll - target[2]) < self.config.TOLERANCE_ROLL
        
        if yaw_ok and pitch_ok and roll_ok:
            elapsed = time.time() - state.pose_start_time
            remaining = max(0, self.config.POSE_DURATION - elapsed)
            
            cv2.putText(display, "✓ POSE CORRECTE", 
                       (w - 280, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Maintenez: {remaining:.1f}s", 
                       (10, 220), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)
            
            if elapsed >= self.config.POSE_DURATION:
                cv2.putText(display, "✓ AUTHENTIFIÉ !", 
                           (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow('Authentification Faciale', display)
                cv2.waitKey(3000)
                logger.info(f"✓ Authentification réussie: {state.authenticated_user}")
                state.reset()
        else:
            cv2.putText(display, "✗ POSE INCORRECTE", 
                       (w - 280, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
            state.pose_start_time = time.time()
    
    def _draw_face_rect(self, frame, rect):
        """Dessine un rectangle moderne autour du visage"""
        x, y = rect.left(), rect.top()
        w, h = rect.width(), rect.height()
        self._draw_modern_face_box(frame, x, y, w, h, (100, 255, 100), 3)
    
    def _draw_modern_face_box(self, frame, x, y, w, h, color, thickness):
        """Dessine un cadre moderne avec coins stylisés"""
        corner_length = 30
        
        # Coins supérieurs
        cv2.line(frame, (x, y), (x + corner_length, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_length), color, thickness)
        
        cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, thickness)
        
        # Coins inférieurs
        cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, thickness)
        
        cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, thickness)
    
    def _draw_pose_overlay(self, frame, yaw, pitch, roll):
        """Affiche les angles de pose avec design moderne"""
        h, w = frame.shape[:2]
        
        # Panneau latéral droit avec transparence
        overlay = frame.copy()
        panel_w = 220
        cv2.rectangle(overlay, (w - panel_w, 0), (w, 200), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
        
        # Titre
        cv2.putText(frame, "ORIENTATION", 
                   (w - panel_w + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Valeurs avec icônes
        y_pos = 70
        metrics = [
            ("Yaw", yaw, (100, 200, 255)),
            ("Pitch", pitch, (150, 255, 150)),
            ("Roll", roll, (255, 180, 100))
        ]
        
        for label, value, color in metrics:
            cv2.putText(frame, label, 
                       (w - panel_w + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"{value:+.1f}", 
                       (w - panel_w + 80, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, "deg", 
                       (w - panel_w + 160, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y_pos += 40
    
    def _draw_no_face(self, frame):
        """Affiche un message moderne quand aucun visage n'est détecté"""
        h, w = frame.shape[:2]
        
        # Overlay sombre
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Message centré avec fond
        msg_w, msg_h = 500, 120
        msg_x, msg_y = (w - msg_w) // 2, (h - msg_h) // 2
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (msg_x, msg_y), (msg_x + msg_w, msg_y + msg_h), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay2, 0.9, frame, 0.1, 0)
        cv2.rectangle(frame, (msg_x, msg_y), (msg_x + msg_w, msg_y + msg_h), (255, 100, 100), 2)
        
        # Icône d'alerte (triangle)
        tri_cx, tri_cy = w // 2, msg_y + 40
        tri_pts = np.array([[tri_cx, tri_cy - 20], [tri_cx - 20, tri_cy + 15], [tri_cx + 20, tri_cy + 15]], np.int32)
        cv2.fillPoly(frame, [tri_pts], (255, 100, 100))
        cv2.putText(frame, "!", (tri_cx - 8, tri_cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 3)
        
        # Texte
        cv2.putText(frame, "AUCUN VISAGE DETECTE", 
                   (msg_x + 80, msg_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# ============================================================================
# CLASSE D'ÉTAT
# ============================================================================

class AuthenticationState:
    """Gère l'état du processus d'authentification"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.stage = "IDLE"
        self.authenticated_user = None
        self.required_pose = None
        self.pose_start_time = 0
    
    def set_user(self, username: str, available_poses: List[str]):
        self.authenticated_user = username
        self.required_pose = np.random.choice(available_poses)
        self.stage = "POSE_CHALLENGE"
        self.pose_start_time = time.time()


# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

def main():
    """Point d'entrée principal du programme"""
    config = Config()
    system = FacialRecognitionSystem(config)
    
    print("\n" + "="*60)
    print("   SYSTÈME D'AUTHENTIFICATION FACIALE PAR POSE")
    print("="*60 + "\n")
    
    while True:
        print("\n📋 MENU PRINCIPAL:")
        print("  1. Capturer un nouveau dataset utilisateur")
        print("  2. Lancer l'authentification")
        print("  3. Quitter")
        
        choice = input("\n➤ Votre choix: ").strip()
        
        if choice == "1":
            username = input("Nom d'utilisateur: ").strip()
            num_images = int(input("Images par pose (défaut 10): ") or "10")
            system.capture_dataset(username, num_images)
            
        elif choice == "2":
            encodings, names, poses = system.load_dataset()
            if len(encodings) > 0:
                print(f"\n👥 {len(names)} utilisateur(s) enregistré(s): {', '.join(names)}")
                system.authenticate(encodings, names, poses)
            else:
                print("\n⚠ Aucun dataset trouvé. Capturez d'abord des images.")
        
        elif choice == "3":
            print("\n👋 Au revoir!")
            break
        else:
            print("\n❌ Choix invalide")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interruption par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}", exc_info=True)
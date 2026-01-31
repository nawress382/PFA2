#!/usr/bin/env python3
"""
Utilitaire de gestion du cache facial
- Régénérer le cache
- Afficher l'état du cache
- Supprimer le cache
"""

import sys
import os
from face_service import load_dataset_encodings, _clear_cache, _get_dataset_hash, CACHE_FILE

def show_cache_status():
    """Afficher l'état du cache."""
    print("\n" + "="*60)
    print("ÉTAT DU CACHE FACIAL")
    print("="*60)
    
    if os.path.exists(CACHE_FILE):
        size_bytes = os.path.getsize(CACHE_FILE)
        size_kb = size_bytes / 1024
        print(f"✅ Cache trouvé: {CACHE_FILE}")
        print(f"   Taille: {size_kb:.2f} KB")
    else:
        print(f"❌ Pas de cache trouvé")
    
    dataset_hash = _get_dataset_hash("dataset")
    print(f"\n📂 Dataset hash: {dataset_hash}")
    
def rebuild_cache():
    """Régénérer complètement le cache."""
    print("\n" + "="*60)
    print("RÉGÉNÉRATION DU CACHE")
    print("="*60)
    
    # Supprimer l'ancien cache
    _clear_cache()
    
    # Charger le dataset (force regeneration)
    encodings, names, poses = load_dataset_encodings("dataset", use_cache=False)
    
    if len(encodings) == 0:
        print("❌ Aucun encodage généré!")
        return False
    
    print(f"\n✅ Cache régénéré avec succès!")
    print(f"\n📊 STATISTIQUES:")
    print(f"   • Utilisateurs: {len(names)}")
    print(f"   • Encodages: {len(encodings)}")
    
    for user in names:
        print(f"\n   👤 {user}")
        user_poses = poses.get(user, {})
        for pose_name, angles in user_poses.items():
            print(f"      • {pose_name}: yaw={angles[0]:.1f}°, pitch={angles[1]:.1f}°, roll={angles[2]:.1f}°")
    
    return True

def clear_cache():
    """Supprimer le cache."""
    print("\n" + "="*60)
    print("SUPPRESSION DU CACHE")
    print("="*60)
    _clear_cache()
    print("✅ Cache supprimé")

def main():
    if len(sys.argv) < 2:
        print(f"\nUsage: python {sys.argv[0]} <command>")
        print(f"\nCommandes:")
        print(f"  status   - Afficher l'état du cache")
        print(f"  rebuild  - Régénérer le cache (vide et reconstruit)")
        print(f"  clear    - Supprimer le cache\n")
        return
    
    command = sys.argv[1].lower()
    
    if command == "status":
        show_cache_status()
    elif command == "rebuild":
        rebuild_cache()
    elif command == "clear":
        clear_cache()
    else:
        print(f"❌ Commande inconnue: {command}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test authentication flow with detailed diagnostics.
Shows angle mismatches frame-by-frame.
"""

import sys
from face_service import load_dataset_encodings, authenticate_sequence

def main():
    print("\n" + "="*70)
    print("AUTHENTIFICATION AVEC DIAGNOSTIQUES DETAILLES")
    print("="*70)
    
    # Load dataset
    encodings, names, poses = load_dataset_encodings()
    
    if not names or len(encodings) == 0:
        print("Aucun utilisateur enregistré dans le dataset!")
        return
    
    print(f"\nUtilisateurs disponibles: {names}")
    print(f"Total encodages: {len(encodings)}")
    
    for user, user_poses in poses.items():
        print(f"\n{user}:")
        for pose_name, (yaw, pitch, roll) in user_poses.items():
            print(f"  {pose_name:10s} : yaw={yaw:6.1f}°  pitch={pitch:6.1f}°  roll={roll:6.1f}°")
    
    print("\n" + "-"*70)
    print("Démarrage de l'authentification avec diagnostic en temps réel...")
    print("Regardez les angles affichés ci-dessous pendant que vous faites les poses.")
    print("-"*70 + "\n")
    
    def status_cb(msg):
        print(f"[STATUS] {msg}")
    
    def progress_cb(val):
        pass  # Don't spam progress
    
    def frame_cb(frame):
        pass  # Ignore frames in this test
    
    try:
        result = authenticate_sequence(
            encodings,
            names,
            poses,
            camera_index=0,
            total_challenges=5,
            challenge_timeout=15,  # longer timeout for diagnostic
            progress_callback=progress_cb,
            status_callback=status_cb,
            frame_callback=frame_cb,
            debug_poses=True,  # ENABLE DETAILED DIAGNOSTICS
        )
        
        print("\n" + "="*70)
        print("RESULTAT AUTHENTIFICATION:")
        print("="*70)
        print(result)
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

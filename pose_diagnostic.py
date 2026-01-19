#!/usr/bin/env python3
"""
Diagnostic script: Show learned poses and monitor real-time pose angles during capture.
This helps debug authentication failures on left/right poses.
"""

import cv2
import dlib
import numpy as np
import os
from face_service import (
    _ensure_models,
    estimate_pose,
    load_dataset_encodings,
    _detector,
    _predictor,
)

def show_pose_diagnostics():
    """Print all learned poses with their angles."""
    print("\n" + "="*70)
    print("DIAGNOSTIC: POSES APPRISES DANS LE DATASET")
    print("="*70)
    
    encodings, names, poses = load_dataset_encodings()
    
    if not names:
        print("Aucun utilisateur dans le dataset!")
        return
    
    for user_name in names:
        user_poses = poses.get(user_name, {})
        print(f"\n{user_name}:")
        for pose_key, (yaw, pitch, roll) in user_poses.items():
            print(f"  {pose_key:10s} : yaw={yaw:6.1f}°  pitch={pitch:6.1f}°  roll={roll:6.1f}°")
    
    print("\n" + "="*70)
    print("TOLERANCES ACTUELLES (utilisées pour l'authentification):")
    print("="*70)
    tolerances = {
        "frontale": (15, 15, 15),
        "gauche": (25, 15, 15),  # yaw wider for side poses
        "droite": (25, 15, 15),  # yaw wider for side poses
        "haut": (15, 20, 15),    # pitch wider for up
        "bas": (15, 20, 15),     # pitch wider for down
    }
    for pose_name, (tol_yaw, tol_pitch, tol_roll) in tolerances.items():
        print(f"  {pose_name:10s} : tol_yaw={tol_yaw}°  tol_pitch={tol_pitch}°  tol_roll={tol_roll}°")
    
    print()

def real_time_pose_monitor():
    """Capture from camera and show real-time pose angles vs learned values."""
    _ensure_models()
    
    encodings, names, poses_dict = load_dataset_encodings()
    if not names:
        print("Aucun utilisateur enregistré - impossible de continuer.")
        return
    
    # Use first user (or let user select)
    user_name = names[0]
    target_poses = poses_dict.get(user_name, {})
    
    print(f"\nMonitoring poses for: {user_name}")
    print(f"Available poses: {list(target_poses.keys())}")
    print("Press 'q' to quit, or 'f/g/d/u/b' to switch pose (frontale/gauche/droite/haut/bas)\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera!")
        return
    
    current_pose = "frontale"
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect face
            small = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = _detector(gray, 1)
            
            display = frame.copy()
            
            if len(faces) > 0:
                face = faces[0]
                rect_orig = dlib.rectangle(
                    int(face.left() / 0.75), int(face.top() / 0.75),
                    int(face.right() / 0.75), int(face.bottom() / 0.75)
                )
                landmarks = _predictor(frame, rect_orig)
                
                # Estimate current pose
                yaw, pitch, roll, _, _ = estimate_pose(landmarks, frame.shape[1], frame.shape[0])
                
                # Get target pose
                target = target_poses.get(current_pose)
                if target:
                    tol_dict = {
                        "frontale": (15, 15, 15),
                        "gauche": (25, 15, 15),
                        "droite": (25, 15, 15),
                        "haut": (15, 20, 15),
                        "bas": (15, 20, 15),
                    }
                    tol_yaw, tol_pitch, tol_roll = tol_dict.get(current_pose, (15, 15, 15))
                    
                    yaw_err = abs(yaw - target[0])
                    pitch_err = abs(pitch - target[1])
                    roll_err = abs(roll - target[2])
                    
                    yaw_match = yaw_err < tol_yaw
                    pitch_match = pitch_err < tol_pitch
                    roll_match = roll_err < tol_roll
                    
                    # Show text on frame
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    color_ok = (0, 255, 0)
                    color_bad = (0, 0, 255)
                    
                    y_pos = 30
                    cv2.putText(display, f"Pose cible: {current_pose}", (10, y_pos), font, 1, (255, 255, 255), 2)
                    
                    y_pos += 40
                    yaw_color = color_ok if yaw_match else color_bad
                    cv2.putText(display, f"YAW:   {yaw:6.1f}° -> {target[0]:6.1f}° (err={yaw_err:5.1f}° tol={tol_yaw}°)", 
                                (10, y_pos), font, 0.7, yaw_color, 2)
                    
                    y_pos += 30
                    pitch_color = color_ok if pitch_match else color_bad
                    cv2.putText(display, f"PITCH: {pitch:6.1f}° -> {target[1]:6.1f}° (err={pitch_err:5.1f}° tol={tol_pitch}°)", 
                                (10, y_pos), font, 0.7, pitch_color, 2)
                    
                    y_pos += 30
                    roll_color = color_ok if roll_match else color_bad
                    cv2.putText(display, f"ROLL:  {roll:6.1f}° -> {target[2]:6.1f}° (err={roll_err:5.1f}° tol={tol_roll}°)", 
                                (10, y_pos), font, 0.7, roll_color, 2)
                    
                    y_pos += 40
                    status = "✓ MATCH" if (yaw_match and pitch_match and roll_match) else "✗ NO MATCH"
                    status_color = (0, 255, 0) if (yaw_match and pitch_match and roll_match) else (0, 0, 255)
                    cv2.putText(display, status, (10, y_pos), font, 1.2, status_color, 3)
            
            cv2.imshow("Pose Diagnostic", display)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                current_pose = "frontale"
            elif key == ord('g'):
                current_pose = "gauche"
            elif key == ord('d'):
                current_pose = "droite"
            elif key == ord('u'):
                current_pose = "haut"
            elif key == ord('b'):
                current_pose = "bas"
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    show_pose_diagnostics()
    print("\nPress any key to start real-time pose monitoring...")
    input()
    real_time_pose_monitor()

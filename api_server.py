import base64
import io
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict
import boto3
import cv2
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import JSONResponse
from security_utils import encrypt_aes_512, upload_to_s3
import os

import face_service
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('FaceAuthTasks')
app = FastAPI(title="Face Auth Service")

# Simple in-memory task manager. Not persistent — fine for local usage.
TASKS: Dict[str, Dict[str, Any]] = {}
_EXECUTOR = ThreadPoolExecutor(max_workers=2)
_LOCK = threading.Lock()
def migrate_user_to_cloud(user_name: str):
    bucket_name = "ton-nom-de-bucket-ici" # <--- METS TON NOM DE BUCKET ICI
    
    # On imagine qu'on migre le fichier d'encodings ou une image
    file_path = f"dataset/{user_name}/face_data.pkl" # ou le chemin de ton choix
    
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
            
        # 1. Chiffrement AES-512 (Double AES-256)
        payload, signature = encrypt_aes_512(data)
        
        # 2. Migration vers S3
        s3_name = f"cloud_storage/{user_name}_secure_data.bin"
        success = upload_to_s3(payload, signature, bucket_name, s3_name)
        
        return success
    return False

def _encode_frame_to_b64(frame_bgr) -> str:
    """Encode BGR frame (numpy) to JPEG-base64 string."""
    try:
        ret, buf = cv2.imencode('.jpg', frame_bgr)
        if not ret:
            return ""
        b = base64.b64encode(buf.tobytes()).decode('ascii')
        return b
    except Exception:
        return ""


def _update_task(task_id: str, **kwargs):
    # 1. On garde la mise à jour locale (optionnel, pour le debug)
    with _LOCK:
        TASKS.setdefault(task_id, {})
        TASKS[task_id].update(kwargs)
    
    # 2. On envoie les données vers AWS DynamoDB
    # On transforme les données pour qu'elles plaisent à DynamoDB
    try:
        status = str(kwargs.get('status', TASKS[task_id].get('status', '')))
        progress = int(kwargs.get('progress', TASKS[task_id].get('progress', 0)))
        
        table.put_item(
            Item={
                'task_id': task_id,
                'status': status,
                'progress': progress,
                # On peut ajouter d'autres infos si besoin
            }
        )
        print(f"Cloud: Task {task_id} mise à jour sur AWS.")
    except Exception as e:
        print(f"Erreur Cloud DynamoDB: {e}")

def _capture_worker(task_id: str, user_name: str, num_images_per_pose: int, output_dir: str, camera_index: int):
    try:
        _update_task(task_id, status="starting", progress=0)

        def progress_cb(event, info):
            # capture_images_headless uses ('pose_start'/'image_saved'/'pose_done') style callbacks
            if isinstance(event, str) and event == 'pose_start':
                _update_task(task_id, status=f"pose:{info.get('pose')}")
            elif isinstance(event, str) and event == 'image_saved':
                _update_task(task_id, progress=info.get('count', 0), last_saved=info.get('path'))
            elif isinstance(event, str) and event == 'pose_done':
                _update_task(task_id, status=f"pose_done:{info.get('pose')}")

        result = face_service.capture_images_headless(
            user_name,
            num_images_per_pose=num_images_per_pose,
            output_dir=output_dir,
            camera_index=camera_index,
            progress_callback=progress_cb,
        )# Une fois la capture finie, on lance la migration sécurisée
        _update_task(task_id, status="migrating_to_cloud")
        migration_ok = migrate_user_to_cloud(user_name)
        
        if migration_ok:
            _update_task(task_id, status="finished_and_migrated", progress=100)
        else:
            _update_task(task_id, status="finished_local_only", error="Migration failed")
            
        _update_task(task_id, status="finished", result=result, progress=100)
    except Exception as e:
        _update_task(task_id, status="error", error=str(e))


def _auth_worker(task_id: str, camera_index: int, total_challenges: int, challenge_timeout: int):
    try:
        _update_task(task_id, status="starting", progress=0)
        # load dataset encodings from disk
        encodings, names, poses = face_service.load_dataset_encodings()
        if len(encodings) == 0 or not names:
            _update_task(task_id, status="error", error="No known encodings in dataset")
            return

        def status_cb(text: str):
            _update_task(task_id, status=text)

        def progress_cb(val: int):
            _update_task(task_id, progress=int(val))

        def frame_cb(frame_bgr):
            # store a small preview (base64)
            b64 = _encode_frame_to_b64(frame_bgr)
            _update_task(task_id, last_frame=b64)

        result = face_service.authenticate_sequence(
            encodings,
            names,
            poses,
            camera_index=camera_index,
            total_challenges=total_challenges,
            challenge_timeout=challenge_timeout,
            progress_callback=progress_cb,
            status_callback=status_cb,
            frame_callback=frame_cb,
        )
        _update_task(task_id, status="finished", result=result, progress=100)
    except Exception as e:
        _update_task(task_id, status="error", error=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/capture")
def start_capture(user_name: str, num_images_per_pose: int = 40, output_dir: str = "dataset", camera_index: int = 0):
    """Start headless capture for a named user. Returns task_id to poll status."""
    task_id = str(uuid.uuid4())
    with _LOCK:
        TASKS[task_id] = {"status": "queued", "progress": 0}
    _EXECUTOR.submit(_capture_worker, task_id, user_name, num_images_per_pose, output_dir, camera_index)
    return {"task_id": task_id}


@app.post("/authenticate")
def start_authenticate(camera_index: int = 0, total_challenges: int = 5, challenge_timeout: int = 12):
    """Start an authentication run. Returns task_id to poll status and retrieve result when done."""
    task_id = str(uuid.uuid4())
    with _LOCK:
        TASKS[task_id] = {"status": "queued", "progress": 0}
    _EXECUTOR.submit(_auth_worker, task_id, camera_index, total_challenges, challenge_timeout)
    return {"task_id": task_id}


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    # On va chercher l'info sur AWS DynamoDB
    response = table.get_item(Key={'task_id': task_id})
    item = response.get('Item')
    
    if not item:
        return JSONResponse(status_code=404, content={"error": "Task not found on AWS"})
    
    return item

@app.get("/dataset")
def dataset_list():
    enc, names, poses = face_service.load_dataset_encodings()
    return {"users": names, "poses": poses}


if __name__ == "__main__":
    print("Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload")

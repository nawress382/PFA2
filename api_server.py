import base64
import io
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import cv2
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import JSONResponse

import face_service

app = FastAPI(title="Face Auth Service")

# Simple in-memory task manager. Not persistent — fine for local usage.
TASKS: Dict[str, Dict[str, Any]] = {}
_EXECUTOR = ThreadPoolExecutor(max_workers=2)
_LOCK = threading.Lock()


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
    with _LOCK:
        TASKS.setdefault(task_id, {})
        TASKS[task_id].update(kwargs)


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
        )
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
    t = TASKS.get(task_id)
    if t is None:
        return JSONResponse(status_code=404, content={"error": "task not found"})
    return t


@app.get("/dataset")
def dataset_list():
    enc, names, poses = face_service.load_dataset_encodings()
    return {"users": names, "poses": poses}


if __name__ == "__main__":
    print("Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload")

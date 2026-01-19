# Face Auth API

This repository contains a PyQt-based face authentication GUI and a small FastAPI wrapper to run headless capture & authentication flows.

Files added by the API work:

- `api_server.py` — FastAPI service that exposes endpoints to start capture and authentication tasks and poll task status.
- `requirements.txt` — project dependencies (may already exist; ensure it contains `fastapi` and `uvicorn[standard]`).

How to run the API server (PowerShell):

1. Create a virtual environment and activate it (optional but recommended):

   python -m venv .venv; .\.venv\Scripts\Activate.ps1

2. Install dependencies:

   pip install -r requirements.txt

3. Start the server with Uvicorn:

   # development mode with auto-reload
   uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:

- GET /health — simple health check
- POST /capture?user_name=alice — start a background capture job; returns {"task_id": "..."}
- POST /authenticate — start a background authentication job; returns {"task_id": "..."}
- GET /tasks/{task_id} — poll the task status/result. The task object will contain fields like `status`, `progress`, `last_frame` (base64 jpeg), `result`, or `error`.
- GET /dataset — returns loaded dataset summary (users, poses)

Notes / limitations:

- This is a simple in-memory task manager for local / demo use. It is not production-ready.
- `dlib` and the model files (shape predictor & face recognition model) must be available where `face_service.py` expects them.
- Camera access is done directly from the server process. If you also run the GUI that uses the camera, you may need to ensure they don't conflict.

If you want, I can:
- Add WebSocket / Server-Sent Events for live frame streaming.
- Persist task state or integrate with a job queue (Redis / RQ / Celery).
- Add an example client (curl / small web page) to illustrate usage.

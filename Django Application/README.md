# Django Application

This folder contains the Django app used for the fake video content detection project.

It is a local web application that loads trained `.pt` model files, processes uploaded videos, and shows a real/fake prediction with confidence.

## What is included

- Django project settings and URL routing
- Upload form and prediction views
- HTML templates for the home, prediction, about, and CUDA fallback pages
- Trained model files in `models/`
- Static assets and template files
- Local upload folders for videos and generated images

## Current behavior

The app:

1. Loads the available model files from `models/`
2. Shows them in the upload form
3. Accepts video uploads such as mp4, avi, mkv, mov, webm, gif, 3gp, and flv
4. Extracts frames from the uploaded video
5. Uses face detection when `face_recognition` is installed
6. Loads the selected PyTorch model
7. Returns a prediction and confidence score

## Requirements

- Python 3.11 to 3.13
- Django 5.0.6
- NumPy
- OpenCV
- Pillow
- PyTorch
- Optional: `face_recognition`, `matplotlib`

## Local setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

## Project structure

### Main files

- `manage.py` — Django entry point
- `db.sqlite3` — local development database
- `requirements.txt` — Python dependencies

### App files

- `ml_app/views.py` — upload handling and prediction logic
- `ml_app/forms.py` — upload form
- `ml_app/urls.py` — app URLs
- `ml_app/templates/` — app templates

### Project settings

- `project_settings/settings.py` — Django settings
- `project_settings/urls.py` — root URL configuration
- `project_settings/wsgi.py` — WSGI entry point
- `project_settings/asgi.py` — ASGI entry point

### Model files

- `models/model_90_acc_20_frames_FF_data.pt`
- `models/model_95_acc_40_frames_FF_data.pt`
- `models/model_97_acc_60_frames_FF_data.pt`
- `models/model_97_acc_80_frames_FF_data.pt`
- `models/model_97_acc_100_frames_FF_data.pt`

## Routes

The Django app exposes these routes:

- `/` — home page
- `/about/` — about page
- `/predict/` — prediction result page
- `/cuda_full/` — CUDA memory fallback page

## Configuration notes

The current settings are development-oriented:

- `DEBUG = True`
- SQLite database
- Media uploads stored locally
- Static assets served from the project directories
- Upload size limit set in `settings.py`

## Useful commands

```bash
python manage.py migrate
python manage.py makemigrations
python manage.py runserver
python manage.py createsuperuser
```

## Notes

- The project does not expose a dedicated REST API in the current Django URL configuration.
- Docker-related files exist in the app folder, but the working local setup is the standard Django development server.
- If PyTorch is unavailable, prediction will not run.
  

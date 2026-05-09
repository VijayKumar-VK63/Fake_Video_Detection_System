# Fake Video Content Detection System

Deepfake detection web application built with Django and a ResNet50 + LSTM inference pipeline.

This repository contains the web app, model files, templates, static assets, and local development configuration used to upload a video and generate a real/fake prediction.

## What the project does

- Upload a video through the Django web interface
- Select from the available trained model files
- Extract frames and optionally crop faces when `face_recognition` is available
- Run inference with PyTorch and OpenCV
- Show the prediction, confidence, and generated preview images

## Current stack

- Django web application
- SQLite database for development
- PyTorch-based model loading and prediction
- OpenCV for video frame handling
- Optional `face_recognition` and `matplotlib` support

## Repository layout

The main application lives in [Django Application/](Django%20Application/):

- `manage.py` — Django entry point
- `ml_app/` — application logic, forms, views, and templates
- `project_settings/` — Django settings and URL routing
- `models/` — trained `.pt` model files
- `static/` — CSS, JavaScript, and assets
- `templates/` — shared site templates
- `uploaded_videos/` — uploaded video files during local use
- `uploaded_images/` — generated preview and cropped images

## Available model files

The app currently includes these trained model files:

- `model_90_acc_20_frames_FF_data.pt`
- `model_95_acc_40_frames_FF_data.pt`
- `model_97_acc_60_frames_FF_data.pt`
- `model_97_acc_80_frames_FF_data.pt`
- `model_97_acc_100_frames_FF_data.pt`

The app builds the model dropdown from files in [Django Application/models/](Django%20Application/models/).

## Supported pages

- `/` — home page with upload form
- `/about/` — project information page
- `/predict/` — prediction results page after upload
- `/cuda_full/` — fallback page for CUDA memory issues

## Requirements

- Python 3.11 to 3.13
- Django 5.0.6
- NumPy
- OpenCV
- Pillow
- PyTorch
- Optional: `face_recognition`, `matplotlib`

## Local setup

1. Open the [Django Application/](Django%20Application/) folder.
2. Create and activate a virtual environment.
3. Install dependencies from `requirements.txt`.
4. Run migrations.
5. Start the development server.

Typical Windows commands:

```bash
cd "Django Application"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

## How inference works

The application:

1. Reads the uploaded video
2. Extracts the first frames up to the selected sequence length
3. Crops the first detected face when face detection is available
4. Loads the selected `.pt` model
5. Produces a real/fake result with confidence

If face detection is not available, the app still processes the video frames.
If PyTorch is not available, the prediction page shows an error message instead of crashing.

## File outputs

During prediction, the app writes generated images to [Django Application/uploaded_images/](Django%20Application/uploaded_images/) and stores uploaded videos in [Django Application/uploaded_videos/](Django%20Application/uploaded_videos/).

## Notes

- This project is currently set up for local development with `DEBUG = True`.
- The database is SQLite in the current codebase.
- No public REST API is exposed in the Django URLs at this time.
- Docker support exists in the repository through the app-level Dockerfile, but the working setup in this project is the local Django runserver workflow.

## License

MIT License. See [LICENSE](LICENSE).

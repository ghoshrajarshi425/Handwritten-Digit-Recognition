# Handwritten Digit Recognition (MNIST) — Flask + CNN

This small project trains (or loads) a CNN on MNIST and serves a simple web page where you can upload an image of a handwritten digit (0–9) to get a prediction.

## Quick start (Windows / PowerShell)

1. Create a new folder and copy these files (or clone the repo): `app.py`, `requirements.txt`, `Procfile`, `.gitignore`.
2. Open VS Code terminal in the project folder and install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
python app.py
```

On first run the script will train a small CNN on MNIST (takes a few minutes). The trained model will be saved as `mnist_cnn_model.h5` in the project folder. Subsequent runs will load the saved model instantly.

When the server is running, open http://127.0.0.1:5000 in your browser and upload a handwritten digit image (.png/.jpg).

## Files

- `app.py` — Flask webserver, model training/loading, upload & predict routes.
- `requirements.txt` — Python dependencies.
- `.gitignore` — ignores model file, __pycache__, env files.
- `Procfile` — for Heroku/Render style deployments (uses Gunicorn).
- `runtime.txt` — optional Python runtime hint for some PaaS.

## Deploying from GitHub

1. Create a new GitHub repository and push this folder as the repository root.
2. You can deploy to many platforms directly from GitHub (Render, Heroku, Railway). The `Procfile` uses `gunicorn app:app`.

Notes:
- TensorFlow is large; some PaaS providers may have limits or require special buildpacks. If you want a managed experience, consider using a container (Docker) or a provider that supports large builds.
- For demo/demo-only use it's fine to run it locally.

## Optional improvements

- Add client-side image pre-processing UI (canvas) so users can draw digits in the browser.
- Add small test(s) for the prediction function.

Enjoy! If you'd like, I can add a sample digit image, a simple Dockerfile, or a GitHub Actions workflow to build & push a Docker image.

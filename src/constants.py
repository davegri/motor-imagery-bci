import os

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

MEDIA_DIR = os.path.join(ROOT_DIR, "media")
IMAGES_DIR = os.path.join(MEDIA_DIR, "images")
AUDIO_DIR = os.path.join(MEDIA_DIR, "audio")
TEXT_DIR = os.path.join(MEDIA_DIR, "text")

RECORDINGS_DIR = os.path.join(ROOT_DIR, "recordings")
RECORDING_PARAMS_PATH = os.path.join(ROOT_DIR, "recording_params.json")

PIPELINES_DIR = os.path.join(ROOT_DIR, "pipelines")
HYPERPARAMS_DIR = os.path.join(ROOT_DIR, "hyperparams")

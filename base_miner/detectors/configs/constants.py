import os

CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.abspath(os.path.join(CONFIGS_DIR, "..")) 
WEIGHTS_DIR = os.path.join(BASE_PATH, "weights")

CONFIG_PATHS = {
    "Roadwork": os.path.join(CONFIGS_DIR, "roadwork.yaml"),
}

HF_REPOS = {
    "Roadwork": os.getenv("MODEL_URL", "")
}

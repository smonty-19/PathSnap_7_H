"""API settings (paths). Override with environment variables.

PATHSNAP_EVAL_DIR   folder of evaluator trips (one sub-folder per trip). Default: data/raw/student_test (rehearsal).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(ROOT, "data", "raw", "student_test"))
MODEL_PATH = os.environ.get("PATHSNAP_MODEL", os.path.join(ROOT, "artifacts", "final_model.joblib"))
META_PATH = os.environ.get("PATHSNAP_META", os.path.join(ROOT, "artifacts", "model_meta.json"))

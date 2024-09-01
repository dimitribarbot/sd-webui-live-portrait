from internal_liveportrait.utils_base import *
from pathlib import Path
import os

from modules.modelloader import load_file_from_url
try:
    from modules.paths_internal import models_path
except Exception:
    try:
        from modules.paths import models_path
    except Exception:
        models_path = os.path.abspath("models")

repo_root = Path(__file__).parent.parent


def get_xpose_lib_dir():
    return os.path.join(repo_root, "liveportrait", "utils", "dependencies", "XPose", "models", "UniPose", "ops", "lib")


def has_xpose_lib():
    xpose_lib_dir = get_xpose_lib_dir()
    return os.path.exists(xpose_lib_dir) and len(os.listdir(xpose_lib_dir)) > 0


def del_xpose_lib_dir():
    import shutil
    xpose_lib_dir = get_xpose_lib_dir()
    shutil.rmtree(xpose_lib_dir, ignore_errors=True)


def download_models(model_root, model_urls):
    if not os.path.exists(model_root):
        os.makedirs(model_root, exist_ok=True)
    
    for local_file, url in model_urls:
        local_path = os.path.join(model_root, local_file)
        if not os.path.exists(local_path):
            load_file_from_url(url, model_dir=model_root)


def download_insightface_models():
    """
    Downloading insightface models from huggingface.
    """
    model_root = os.path.join(models_path, "insightface", "models", "buffalo_l")
    model_urls = (
        ("det_10g.onnx", "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/insightface/models/buffalo_l/det_10g.onnx"),
        ("2d106det.onnx", "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/insightface/models/buffalo_l/2d106det.onnx"),
    )
    download_models(model_root, model_urls)


def download_liveportrait_landmark_model():
    """
    Downloading liveportrait landmark model from huggingface.
    """
    model_root = os.path.join(models_path, "liveportrait")
    model_urls = (
        ("landmark.onnx", "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/landmark.onnx"),
    )
    download_models(model_root, model_urls)


def download_liveportrait_base_models():
    """
    Downloading liveportrait base models from huggingface.
    """
    model_root = os.path.join(models_path, "liveportrait", "base_models")
    model_urls = (
        ("appearance_feature_extractor.safetensors", "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/appearance_feature_extractor.safetensors"),
        ("motion_extractor.safetensors", "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/motion_extractor.safetensors"),
        ("spade_generator.safetensors", "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/spade_generator.safetensors"),
        ("warping_module.safetensors", "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/warping_module.safetensors"),
    )
    download_models(model_root, model_urls)


def download_liveportrait_retargeting_models():
    """
    Downloading liveportrait retargeting models from huggingface.
    """
    model_root = os.path.join(models_path, "liveportrait", "retargeting_models")
    model_urls = (
        ("stitching_retargeting_module.safetensors", "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/stitching_retargeting_module.safetensors"),
    )
    download_models(model_root, model_urls)


def download_liveportrait_models():
    """
    Downloading liveportrait models.
    """
    download_liveportrait_landmark_model()
    download_liveportrait_base_models()
    download_liveportrait_retargeting_models()


def download_liveportrait_animals_xpose_model():
    """
    Downloading liveportrait animals xpose model from huggingface.
    """
    model_root = os.path.join(models_path, "liveportrait_animals")
    model_urls = (
        ("xpose.pth", "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/liveportrait_animals/xpose.pth"),
    )
    download_models(model_root, model_urls)


def download_liveportrait_animals_base_models():
    """
    Downloading liveportrait animals base models from huggingface.
    """
    model_root = os.path.join(models_path, "liveportrait_animals", "base_models")
    model_urls = (
        ("appearance_feature_extractor.safetensors", "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/animal/appearance_feature_extractor.safetensors"),
        ("motion_extractor.safetensors", "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/animal/motion_extractor.safetensors"),
        ("spade_generator.safetensors", "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/animal/spade_generator.safetensors"),
        ("warping_module.safetensors", "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/animal/warping_module.safetensors"),
    )
    download_models(model_root, model_urls)


def download_liveportrait_animals_retargeting_models():
    """
    Downloading liveportrait animals retargeting models from huggingface.
    """
    model_root = os.path.join(models_path, "liveportrait_animals", "retargeting_models")
    model_urls = (
        ("stitching_retargeting_module.safetensors", "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/animal/stitching_retargeting_module.safetensors"),
    )
    download_models(model_root, model_urls)


def download_liveportrait_animals_models():
    """
    Downloading liveportrait animals models.
    """
    download_liveportrait_animals_xpose_model()
    download_liveportrait_animals_base_models()
    download_liveportrait_animals_retargeting_models()

import launch
import os, sys
import shutil
from importlib import metadata
from pathlib import Path
from typing import Optional
from packaging.version import parse
from modules.modelloader import load_file_from_url
import subprocess

try:
    from modules.paths_internal import models_path
except:
    try:
        from modules.paths import models_path
    except:
        models_path = os.path.abspath("models")


repo_root = Path(__file__).parent
main_req_file = repo_root / "requirements.txt"


def get_installed_version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except Exception:
        return None


def extract_base_package(package_string: str) -> str:
    base_package = package_string.split("@git")[0]
    return base_package


def install_requirements(req_file):
    with open(req_file) as file:
        for package in file:
            try:
                package = package.strip()
                if "==" in package:
                    package_name, package_version = package.split("==")
                    installed_version = get_installed_version(package_name)
                    if installed_version != package_version:
                        launch.run_pip(
                            f'install -U "{package}"',
                            f"sd-webui-live-portrait requirement: changing {package_name} version from {installed_version} to {package_version}",
                        )
                elif ">=" in package:
                    package_name, package_version = package.split(">=")
                    installed_version = get_installed_version(package_name)
                    if not installed_version or parse(
                        installed_version
                    ) < parse(package_version):
                        launch.run_pip(
                            f'install -U "{package}"',
                            f"sd-webui-live-portrait requirement: changing {package_name} version from {installed_version} to {package_version}",
                        )
                elif "<=" in package:
                    package_name, package_version = package.split("<=")
                    installed_version = get_installed_version(package_name)
                    if not installed_version or parse(
                        installed_version
                    ) > parse(package_version):
                        launch.run_pip(
                            f'install "{package_name}=={package_version}"',
                            f"sd-webui-live-portrait requirement: changing {package_name} version from {installed_version} to {package_version}",
                        )
                elif not launch.is_installed(extract_base_package(package)):
                    launch.run_pip(
                        f'install "{package}"',
                        f"sd-webui-live-portrait requirement: {package}",
                    )
            except Exception as e:
                print(e)
                print(
                    f"Warning: Failed to install {package}, some preprocessors may not work."
                )


def install_onnxruntime():
    """
    Install onnxruntime or onnxruntime-gpu based on the availability of CUDA.
    onnxruntime and onnxruntime-gpu can not be installed together.
    """
    if not launch.is_installed("onnxruntime") and not launch.is_installed("onnxruntime-gpu"):
        import torch.cuda as cuda # torch import head to improve loading time
        onnxruntime = 'onnxruntime-gpu' if cuda.is_available() else 'onnxruntime'
        launch.run_pip(
            f'install {onnxruntime}',
            f"sd-webui-live-portrait requirement: {onnxruntime}",
        )


def install_xpose():
    """
    Install XPose.
    """
    if sys.platform.startswith('darwin'):
        # XPose is incompatible with MacOS
        return
    op_root = os.path.join(repo_root, "liveportrait", "utils", "dependencies", "XPose", "models", "UniPose", "ops")
    op_build = os.path.join(op_root, "build")
    op_lib = os.path.join(op_root, "lib")
    if not os.path.exists(op_build) or len(os.listdir(op_build)) == 0:
        print("Building sd-webui-live-portrait requirement: XPose", flush=True)
        subprocess.run([sys.executable, "setup.py", "build"], cwd=op_root, capture_output=True, shell=False)
    if not os.path.exists(op_lib) or len(os.listdir(op_lib)) == 0:
        print("Installing sd-webui-live-portrait requirement: XPose", flush=True)
        if not os.path.exists(op_lib):
            os.makedirs(op_lib, exist_ok=True)
        lib_src = Path(op_build)
        lib_dst = Path(op_lib)
        for lib_file in lib_src.rglob("*.so"):
            shutil.copy2(lib_file, lib_dst)


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


def download_model_weights():
    download_insightface_models()
    download_liveportrait_models()
    if not sys.platform.startswith('darwin'):
        download_liveportrait_animals_models()


install_requirements(main_req_file)
install_onnxruntime()
install_xpose()
download_model_weights()
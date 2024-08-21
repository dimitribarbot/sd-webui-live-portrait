import os
import sys
from pathlib import Path
import subprocess
import sysconfig
from typing import Optional

from modules.modelloader import load_file_from_url
try:
    from modules.paths_internal import models_path
except:
    try:
        from modules.paths import models_path
    except:
        models_path = os.path.abspath("models")


IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform.startswith('darwin')

# A map keyed by get_platform() return values to values accepted by
# 'vcvarsall.bat'. Always cross-compile from x86 to work with the
# lighter-weight MSVC installs that do not include native 64-bit tools.
PLAT_TO_VCVARS = {
    'win32': 'x86',
    'win-amd64': 'x86_amd64',
    'win-arm32': 'x86_arm',
    'win-arm64': 'x86_arm64',
}


repo_root = Path(__file__).parent.parent


def is_valid_torch_version():
    import torch.cuda as cuda
    if cuda.is_available():
        from torch.version import __version__
        return not __version__.startswith("2.1")
    return False


def has_xpose_lib():
    xpose_lib_dir = os.path.join(repo_root, "liveportrait", "utils", "dependencies", "XPose", "models", "UniPose", "ops", "lib")
    return os.path.exists(xpose_lib_dir) and len(os.listdir(xpose_lib_dir)) > 0


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


def _msvc14_find_vc2019():
    """Inspired from "setuptools/msvc.py", replacing -latest by -version 
    to find the right Visual Studio version compatible with CUDA Toolkit 11.8

    Returns "path" based on the result of invoking vswhere.exe
    If no install is found, returns "None"

    If vswhere.exe is not available, by definition, VS 2019 or VS 2022 < 17.9 is not
    installed.
    """
    root = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles")
    if not root:
        return None

    suitable_components = (
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "Microsoft.VisualStudio.Workload.WDExpress",
    )

    for component in suitable_components:
        path = (
            subprocess.check_output([
                os.path.join(root, "Microsoft Visual Studio", "Installer", "vswhere.exe"),
                "-version",
                "[16.0,17.10)",
                "-prerelease",
                "-requires",
                component,
                "-property",
                "installationPath",
                "-products",
                "*",
            ])
            .decode(encoding="mbcs", errors="strict")
            .strip()
        )

        path = os.path.join(path, "VC", "Auxiliary", "Build")
        if os.path.isdir(path):
            return path

    return None


def _msvc14_find_vcvarsall():
    """Inspired by "setuptools/msvc.py"
    """
    best_dir = _msvc14_find_vc2019()

    if not best_dir:
        return None

    vcvarsall = os.path.join(best_dir, "vcvarsall.bat")
    if not os.path.isfile(vcvarsall):
        return None

    return vcvarsall


def _get_vcvarsall_platform():
    """Inspired by "setuptools/_disutils/_msvccompiler.py"
    """
    return PLAT_TO_VCVARS.get(sysconfig.get_platform())


def _find_cuda_home() -> Optional[str]:
    r'''Inspired by torch.utils.cpp_extension.py
    Finds the CUDA install path.
    '''
    try:
        with open(os.devnull, 'w') as devnull:
            nvcc_paths = subprocess.check_output(['where', 'nvcc'],
                                            stderr=devnull).decode(*('oem',)).rstrip('\r\n').split('\r\n')
            nvcc = [nvcc_path for nvcc_path in nvcc_paths if "v11.8" in nvcc_path and os.path.exists(nvcc_path)]
            if len(nvcc) == 0:
                return None
            return os.path.dirname(os.path.dirname(nvcc[0]))
    except Exception:
        cuda_home = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8'
        if not os.path.exists(cuda_home):
            return None
        return cuda_home
    
    
def get_xpose_build_commands_and_env():
    env = os.environ
    commands = [sys.executable, "setup.py", "build"]

    if not IS_WINDOWS:
        return commands, env
    
    vcvarsall = _msvc14_find_vcvarsall()
    if vcvarsall is None:
        root = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles") or "C:/Program Files"
        msvc_path = os.path.join(root, "Microsoft Visual Studio", "2019", "BuildTools", "VC", "Auxiliary", "Build")
        install_url = "https://learn.microsoft.com/en-us/visualstudio/releases/2019/history#release-dates-and-build-numbers"
        print(f"SD-WEBUI-LIVE-PORTRAIT (WARNING): Expected to find folder such as {msvc_path}. Please check if Microsoft Visual Studio 2019 Build Tools is correctly installed. If not, download 'Build Tools' installer at {install_url}.")
        return commands, env
    
    vc_plat_spec = _get_vcvarsall_platform()
    if vc_plat_spec is None:
        print(f"SD-WEBUI-LIVE-PORTRAIT (WARNING): Your operating system platform is not supported. It must be one of {tuple(PLAT_TO_VCVARS)}.")
        return commands, env
    
    cuda_home = _find_cuda_home()
    if cuda_home is None:
        root = os.environ.get("ProgramFiles") or "C:/Program Files"
        cuda_path = os.path.join(root, "NVIDIA GPU Computing Toolkit", "CUDA", "v11.8")
        install_url = "https://developer.nvidia.com/cuda-11-8-0-download-archive"
        print(f"SD-WEBUI-LIVE-PORTRAIT (WARNING): Expected to find folder such as {cuda_path}. Please check if CUDA Toolkit v11.8 is correctly installed. If not, download it at {install_url}.")
        return commands, env
    
    env = env.copy()
    env["DISTUTILS_USE_SDK"] = "1"
    env["MSSdk"] = "1"
    env["CUDA_HOME"] = cuda_home
    
    commands = [vcvarsall, vc_plat_spec, "&&"] + commands

    return commands, env
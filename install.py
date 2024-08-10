import launch
import os, sys
import shutil
from importlib import metadata
from pathlib import Path
from typing import Optional
from packaging.version import parse
from modules.modelloader import load_file_from_url
import subprocess

from internal_liveportrait.utils import is_valid_cuda_version, isMacOS

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
    if not is_valid_cuda_version() or isMacOS():
        # XPose is incompatible with MacOS or torch version 2.1.x
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


install_requirements(main_req_file)
install_onnxruntime()
install_xpose()
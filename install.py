import launch
import os, sys
import shutil
from importlib import metadata
from pathlib import Path
from typing import Optional
from packaging.version import parse
import subprocess
import tempfile

from internal_liveportrait.utils import is_valid_torch_version, is_mac_os


# Based on https://onnxruntime.ai/docs/reference/compatibility.html#onnx-opset-support
onnx_to_onnx_runtime_versions = {
    "1.16.1": "1.18",
    "1.16.0": "1.18",
    "1.15.0": "1.17",
    "1.14.1": "1.16",
    "1.14.0": "1.15",
    "1.13.1": "1.14",
    "1.13.0": "1.14",
    "1.12.0": "1.13",
    "1.11.0": "1.11",
    "1.10.2": "1.10",
    "1.10.1": "1.10",
    "1.10.0": "1.10"
}


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
        installed_onnx_version = get_installed_version("onnx")
        if installed_onnx_version:
            onnx_version = parse(installed_onnx_version)
            onnxruntime_version = onnx_to_onnx_runtime_versions.get(onnx_version.base_version, None)
        launch.run_pip(
            f'install {f"{onnxruntime}=={onnxruntime_version}" if onnxruntime_version else onnxruntime}',
            f"sd-webui-live-portrait requirement: {onnxruntime}",
        )


def install_xpose():
    """
    Install XPose.
    """
    if not is_valid_torch_version() or is_mac_os():
        # XPose is incompatible with MacOS or torch version 2.1.x
        return
    op_root = os.path.join(repo_root, "liveportrait", "utils", "dependencies", "XPose", "models", "UniPose", "ops")
    op_lib = os.path.join(op_root, "lib")
    if not os.path.exists(op_lib) or len(os.listdir(op_lib)) == 0:
        print("Installing sd-webui-live-portrait requirement: XPose", flush=True)
        if not os.path.exists(op_lib):
            os.makedirs(op_lib, exist_ok=True)
        op_logs = os.path.join(repo_root, "logs")
        if not os.path.exists(op_logs):
            os.makedirs(op_logs, exist_ok=True)
        log_file = os.path.join(op_logs, "xpose.log")
        log_err_file = os.path.join(op_logs, "xpose.err.log")
        with tempfile.TemporaryDirectory() as tmpdirname:
            shutil.copytree(op_root, tmpdirname, dirs_exist_ok=True)
            with open(log_file, 'w') as log_f, open(log_err_file, 'w') as log_err_f:
                result = subprocess.run(
                    [sys.executable, "setup.py", "build"],
                    cwd=tmpdirname,
                    env=os.environ,
                    errors="ignore",
                    stdout=log_f,
                    stderr=log_err_f
                )
                if result.returncode > 0:
                    print("Building of OP file for XPose has failed. Check the log file in the extension's 'logs' folder for more information.")
                    return
            op_build = os.path.join(tmpdirname, "build")
            lib_src = Path(op_build)
            lib_dst = Path(op_lib)
            for lib_file in lib_src.rglob("MultiScaleDeformableAttention*"):
                shutil.copy2(lib_file, lib_dst)


install_requirements(main_req_file)
install_onnxruntime()
install_xpose()
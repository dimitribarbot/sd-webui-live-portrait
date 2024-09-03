import launch
import os
from pathlib import Path
from packaging.version import parse
import tempfile

from internal_liveportrait.utils_base import IS_WINDOWS, IS_MACOS, get_xpose_build_commands_and_env, get_installed_version

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


def get_onnxruntime_version_given_onnx_version():
    installed_onnx_version = get_installed_version("onnx")
    if installed_onnx_version:
        onnx_version = parse(installed_onnx_version)
        onnxruntime_version = onnx_to_onnx_runtime_versions.get(onnx_version.base_version, None)
        return onnxruntime_version
    return None


def are_versions_similar(version_left: str, version_right: str):
    parsed_version_left = parse(version_left)
    parsed_version_right = parse(version_right)
    return parsed_version_left.major == parsed_version_right.major and parsed_version_left.minor == parsed_version_right.minor


def get_onnxruntime_extra_index():
    """
    onnxruntime-gpu requires wheel from a different index for CUDA 12
    https://onnxruntime.ai/docs/install/#install-onnx-runtime-gpu-cuda-12x
    https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/onnxruntime-cuda-12/PyPI/onnxruntime-gpu/overview
    https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/onnxruntime-cuda-12/connect
    """
    import subprocess
    import re
    try:
        if re.search(r'CUDA\s+Version:\s+([0-9.]+)\s*', subprocess.check_output(["nvidia-smi"]).decode()).group(1).startswith('12'):
            return ' --extra-index-url "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"'
    except Exception as e:
        print(f'Unable to get CUDA version: {e}')
    return ''


def install_onnxruntime():
    """
    Install onnxruntime or onnxruntime-gpu based on the availability of CUDA.
    onnxruntime and onnxruntime-gpu can not be installed together.
    """
    onnxruntime_installed_version = get_installed_version("onnxruntime")
    onnxruntime_gpu_installed_version = get_installed_version("onnxruntime-gpu")

    expected_onnxruntime_version = get_onnxruntime_version_given_onnx_version()

    if not onnxruntime_installed_version and not onnxruntime_gpu_installed_version:
        import torch.cuda as cuda  # torch import head to improve loading time

        if cuda.is_available():
            onnxruntime = 'onnxruntime-gpu'
            onnxruntime_package = f"onnxruntime-gpu=={expected_onnxruntime_version}" if expected_onnxruntime_version else onnxruntime
            onnxruntime_package += get_onnxruntime_extra_index()
        else:
            onnxruntime = 'onnxruntime'
            onnxruntime_package = f"onnxruntime=={expected_onnxruntime_version}" if expected_onnxruntime_version else onnxruntime

        launch.run_pip(
            f'install {onnxruntime_package}',
            f"sd-webui-live-portrait requirement: {onnxruntime_package}",
        )
    else:
        if onnxruntime_installed_version and expected_onnxruntime_version \
                and not are_versions_similar(onnxruntime_installed_version, expected_onnxruntime_version):
            onnxruntime_package = f"onnxruntime=={expected_onnxruntime_version}"
            launch.run_pip(
                f'install {onnxruntime_package}',
                f"sd-webui-live-portrait requirement: {onnxruntime_package}",
            )
        if onnxruntime_gpu_installed_version and expected_onnxruntime_version \
                and not are_versions_similar(onnxruntime_gpu_installed_version, expected_onnxruntime_version):
            onnxruntime_gpu_package = f"onnxruntime-gpu=={expected_onnxruntime_version}{get_onnxruntime_extra_index()}"

            launch.run_pip(
                f'install {onnxruntime_gpu_package}',
                f"sd-webui-live-portrait requirement: {onnxruntime_gpu_package}",
            )


def install_xpose():
    """
    Install XPose.
    """
    op_root = os.path.join(repo_root, "liveportrait", "utils", "dependencies", "XPose", "models", "UniPose", "ops")
    op_lib = os.path.join(op_root, "lib")
    if not os.path.exists(op_lib):
        os.makedirs(op_lib, exist_ok=True)
        if IS_MACOS:
            # XPose is incompatible with MacOS
            return
        print("Installing sd-webui-live-portrait requirement: XPose", flush=True)
        import subprocess
        import shutil
        op_logs = os.path.join(repo_root, "logs")
        if not os.path.exists(op_logs):
            os.makedirs(op_logs, exist_ok=True)
        log_file = os.path.join(op_logs, "xpose.log")
        log_err_file = os.path.join(op_logs, "xpose.err.log")
        with tempfile.TemporaryDirectory() as tmpdirname:
            shutil.copytree(op_root, tmpdirname, dirs_exist_ok=True)
            with open(log_file, 'w') as log_f, open(log_err_file, 'w') as log_err_f:
                commands, env = get_xpose_build_commands_and_env()
                result = subprocess.run(
                    commands,
                    cwd=tmpdirname,
                    env=env,
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
            extension = ".pyd" if IS_WINDOWS else ".so"
            for lib_file in lib_src.rglob(f"MultiScaleDeformableAttention*{extension}"):
                shutil.copy2(lib_file, lib_dst)


install_requirements(main_req_file)
install_onnxruntime()
install_xpose()

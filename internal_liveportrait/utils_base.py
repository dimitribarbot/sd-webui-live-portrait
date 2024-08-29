import os
import sys
import subprocess
from typing import Optional


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


def get_installed_version(package: str) -> Optional[str]:
    try:
        from importlib import metadata
        return metadata.version(package)
    except Exception:
        return None


def is_valid_torch_version():
    if get_installed_version("torch").startswith("2.1"):
        return False
    import torch.cuda as cuda
    if cuda.is_available():
        return True
    return False


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
    import sysconfig
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

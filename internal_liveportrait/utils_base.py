import os
import sys
from packaging.version import parse, Version
from typing import Optional, Literal


IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform.startswith('darwin')

SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()

# A map keyed by sysconfig.get_platform() return values to values accepted by 'vcvarsall.bat'.
# Uses the native MSVC host if the host platform would need expensive emulation for x86.
PLAT_TO_VCVARS = {
    'win32': 'x86',
    'win-amd64': 'amd64',
    'win-arm32': 'arm',
    'win-arm64': 'arm64',
}

TARGET_TO_PLAT = {
    'x86': 'win32',
    'x64': 'win-amd64',
    'arm': 'win-arm32',
    'arm64': 'win-arm64',
}


VS_Version = Literal['2019', '2022']


def get_installed_version(package: str) -> Optional[str]:
    try:
        from importlib import metadata
        return metadata.version(package)
    except Exception:
        return None


def _msvc14_find_vc(vs_version: VS_Version):
    """Inspired by "setuptools/_distutils/_msvccompiler.py", replacing -latest by -version
    to find the right Visual Studio version compatible with CUDA Toolkit 11.8

    Returns "path" based on the result of invoking vswhere.exe
    If no install is found, returns "None"

    If vswhere.exe is not available, by definition, VS 2019 or VS 2022 < 17.9 is not
    installed.
    """

    root = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles")
    if not root:
        return None
    
    import subprocess

    version_arguments = ["-version", "[16.0,17.10)"] if vs_version == "2019" else ["-latest"]

    suitable_components = (
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "Microsoft.VisualStudio.Workload.WDExpress",
    )

    for component in suitable_components:
        path = (
            subprocess.check_output([
                os.path.join(root, "Microsoft Visual Studio", "Installer", "vswhere.exe"),
                *version_arguments,
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


def _msvc14_find_vcvarsall(vs_version: VS_Version):
    """Inspired by "setuptools/_distutils/_msvccompiler.py"
    """
    best_dir = _msvc14_find_vc(vs_version)

    if not best_dir:
        return None

    vcvarsall = os.path.join(best_dir, "vcvarsall.bat")
    if not os.path.isfile(vcvarsall):
        return None

    return vcvarsall


def _get_vcvarsall_platform():
    """Inspired by "setuptools/_distutils/_msvccompiler.py"
    """
    import sysconfig
    host_platform = sysconfig.get_platform()
    if host_platform not in PLAT_TO_VCVARS:
        return None
    target = os.environ.get('VSCMD_ARG_TGT_ARCH', '')
    platform = TARGET_TO_PLAT.get(target, host_platform)
    if host_platform != 'win-arm64':
        host_platform = 'win32'
    vc_hp = PLAT_TO_VCVARS[host_platform]
    vc_plat = PLAT_TO_VCVARS[platform]
    return vc_hp if vc_hp == vc_plat else f'{vc_hp}_{vc_plat}'


def _get_torch_cuda_major_version():
    try:
        from torch.version import cuda
        return parse(cuda).major
    except Exception:
        return 11


def _find_cuda_home(torch_cuda_major_version: int) -> Optional[str]:
    r'''Inspired by torch.utils.cpp_extension.py
    Finds the CUDA install path.
    '''
    import subprocess
    try:
        with open(os.devnull, 'w') as devnull:
            nvcc_paths = subprocess.check_output(['where', 'nvcc'],
                                                 stderr=devnull).decode(*SUBPROCESS_DECODE_ARGS).rstrip('\r\n').split('\r\n')
            nvcc = [nvcc_path for nvcc_path in nvcc_paths if f"v{torch_cuda_major_version}." in nvcc_path and os.path.exists(nvcc_path)]
            if len(nvcc) == 0:
                return None
            return os.path.dirname(os.path.dirname(nvcc[0]))
    except Exception:
        import glob
        cuda_homes = glob.glob(f'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v{torch_cuda_major_version}.*')
        if len(cuda_homes) == 0:
            cuda_home = ''
        else:
            cuda_home = cuda_homes[0]
        if not os.path.exists(cuda_home):
            cuda_home = None
        return cuda_home


def _get_cuda_version(cuda_home: str | None) -> Optional[Version]:
    """Inspired by torch.utils.cpp_extension.py
    """
    if not cuda_home:
        return None
    import re
    import subprocess
    try:
        nvcc = os.path.join(cuda_home, 'bin', 'nvcc')
        cuda_version_str = subprocess.check_output([nvcc, '--version']).strip().decode(*SUBPROCESS_DECODE_ARGS)
        cuda_version = re.search(r'release (\d+[.]\d+)', cuda_version_str)
        if cuda_version is None:
            return None
        return parse(cuda_version.group(1))
    except Exception:
        return None

    
def get_xpose_build_commands_and_env():
    env = os.environ
    commands = [sys.executable, "setup.py", "build"]

    if not IS_WINDOWS:
        return commands, env
    
    torch_cuda_major_version = _get_torch_cuda_major_version()
    cuda_home = _find_cuda_home(torch_cuda_major_version)
    cuda_version = _get_cuda_version(cuda_home)
    if cuda_home is None or cuda_version is None:
        root = os.environ.get("ProgramFiles") or "C:/Program Files"
        expected_cuda_version = "v11.8" if torch_cuda_major_version == 11 else "v12.x"
        cuda_path = os.path.join(root, "NVIDIA GPU Computing Toolkit", "CUDA", expected_cuda_version, "bin")
        install_url = "https://developer.nvidia.com/cuda-toolkit-archive"
        print(f"SD-WEBUI-LIVE-PORTRAIT (WARNING): Expected to find folder such as {cuda_path} containing an 'nvcc' binary. Please check if CUDA Toolkit is correctly installed. If not, download a version compatible with your PyTorch installed version ({expected_cuda_version}) at {install_url}.")
        return commands, env
    
    vc_plat_spec = _get_vcvarsall_platform()
    if vc_plat_spec is None:
        print(f"SD-WEBUI-LIVE-PORTRAIT (WARNING): Your operating system platform is not supported. It must be one of {tuple(PLAT_TO_VCVARS)}.")
        return commands, env
    
    vs_version: VS_Version = "2022" if cuda_version.major > 12 and cuda_version.minor > 3 else "2019"
    vcvarsall = _msvc14_find_vcvarsall(vs_version)
    if vcvarsall is None:
        root = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles") or "C:/Program Files"
        msvc_path = os.path.join(root, "Microsoft Visual Studio", "2019", "BuildTools", "VC", "Auxiliary", "Build")
        install_url = "https://learn.microsoft.com/en-us/visualstudio/releases/2019/history#release-dates-and-build-numbers"
        print(f"SD-WEBUI-LIVE-PORTRAIT (WARNING): Expected to find folder such as {msvc_path}. Please check if Microsoft Visual Studio 2019 Build Tools is correctly installed. If not, download 'Build Tools' installer at {install_url}.")
        return commands, env
    
    env = env.copy()
    env["DISTUTILS_USE_SDK"] = "1"
    env["MSSdk"] = "1"
    env["CUDA_HOME"] = cuda_home
    
    commands = [vcvarsall, vc_plat_spec, "&&"] + commands

    return commands, env

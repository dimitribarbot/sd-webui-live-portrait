# Old procedure to install XPose (by downgrading CUDA to v11.8)

XPose, the face detector model used for animal mode, is currently not working with MacOS or non NVIDIA graphic cards.  

You can find here the procedure to install XPose on your computer by downgrading CUDA to v11.8.

## Windows Users

### CUDA Toolkit 11.8

First, you need to install the v11.8 CUDA Toolkit. Go to [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive) page and select your operating system configuration. You can either choose "exe (local)" or "exe (network)" but the latter will be faster as you don't need to install all NVIDIA packages. Download the installer and execute it.

![image](./install-cuda-11.8-0.png)

When asked, choose custom install:

![image](./install-cuda-11.8-1.png)

Then in the following dialog, unselect "Other Components" and "Driver Components":

![image](./install-cuda-11.8-2.png)

Finally, under "CUDA", only select "Development" and "Runtime" and click "Next":

![image](./install-cuda-11.8-3.png)

Click "Next" and "Finish" to end CUDA Toolkit installation.

### Microsoft Visual Studio Build Tools

Then, you need to have the proper installation of the Microsoft Visual Studio Build Tools. Go to [Visual Studio 2019 Release History](https://learn.microsoft.com/en-us/visualstudio/releases/2019/history#release-dates-and-build-numbers) and download the Build Tools corresponding to the latest version:  

![image](./install-msvc-16.x-0.png)

> [!NOTE]
> Note that more recent versions of the Build Tools (Visual Studio 2022 17.10 and above) are not compatible with CUDA Toolkit 11.8. Even if you already have Visual Studio 2022 Build Tools installed you still have to download and install the 2019 version.  

> [!WARNING]
> If you're following this tutorial out of the Automatic1111's WebUI extension installation context, ensure to uninstall all previous installations of Visual Studio 2022 Build Tools 17.10 or newer, otherwise the procedure will fail.  

In the installation dialog, select "Desktop development with C++" as shown in the image below (ensure that the version to be installed is the correct one) and click "Install":

![image](./install-msvc-16.x-1.png)

At the end of the installation procedure, you should see a screen like the following:

![image](./install-msvc-16.x-2.png)

### Automatic1111's SD WebUI

Open the `stable-diffusion-webui/webui-user.bat` file and make the following changes:

If you're using `xformers`, adjust or add the following lines (if you're not using `xformers`, remove the `--xformers` flag in `COMMANDLINE_ARGS` and remove the `XFORMERS_PACKAGE` line):
```
set COMMANDLINE_ARGS=--skip-version-check --xformers
set TORCH_COMMAND=pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
set XFORMERS_PACKAGE=xformers==0.0.22
```

As we have downgraded the pytorch version, to avoid unnecessary warnings at each launch of Automatic1111's SD WebUI, we add the `--skip-version-check` flag to the command line arguments.

After these modifications, close Stable Diffusion WebUI if not done and restart it using the flags `--reinstall-torch` and `--reinstall-xformers` (if you're using `xformers`). These flags can then be removed for subsequent launches of Automatic1111's SD WebUI.

If everything went well, you should be able to use animal mode in the `Live Portrait` tab.

## Linux Users

### GCC

Verify that you have `gcc` correctly installed by running the following command in a terminal:
```
gcc --version
```
If an error message displays, you need to install the development tools from your Linux distribution or obtain a version of gcc and its accompanying toolchain from the Web.

### CUDA Toolkit 11.8

Then, you need to install the v11.8 CUDA Toolkit. Go to [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive) page and select your operating system configuration. You can choose any of the "Installer Type" but note that you only need the CUDA Toolkit SDK to be installed, you can skip the driver installation if you've already done it.

![image](./install-cuda-11.8-4.png)

At the end of the installation procedure, add the correct version of CUDA Toolkit to your PATH environment variable as described [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#environment-setup) by replacing latest version of CUDA with 11.8 in folder paths: `/usr/local/cuda-11.8/`.

### Automatic1111's SD WebUI

Open the `stable-diffusion-webui/webui-user.sh` file and make the following changes:

If you're using `xformers`, adjust or add the following lines (if you're not using `xformers`, remove the `--xformers` flag in `COMMANDLINE_ARGS` and remove the `XFORMERS_PACKAGE` line):
```
export COMMANDLINE_ARGS="--skip-version-check --xformers"
export TORCH_COMMAND="pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118"
export XFORMERS_PACKAGE="xformers==0.0.22"
```

As we have downgraded the pytorch version, to avoid unnecessary warnings at each launch of Automatic1111's SD WebUI, we add the `--skip-version-check` flag to the command line arguments.

After these modifications, close Stable Diffusion WebUI if not done and restart it using the flags `--reinstall-torch` and `--reinstall-xformers` (if you're using `xformers`). These flags can then be removed for subsequent launches of Automatic1111's SD WebUI.

If everything went well, you should be able to use animal mode in the `Live Portrait` tab.

## WSL Users (Windows Subsystem for Linux)

The installation procedure is the same as for Linux users, except that you need to select `WSL-Ubuntu` in the CUDA Toolkit 11.8 step. More information on how to install CUDA when using WSL can be found [here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

> [!Note]
> Verify the CUDA version used by your system by running the `nvidia-smi` command in a terminal. If you have CUDA 12.x as screenshot below, you will need both CUDA Toolkit 11.8 and CUDA Toolkit 12.x to be installed in WSL. However, your environment variables should still points to the 11.8 CUDA version. Not installing CUDA Toolkit 12.x may lead to errors with onnxruntime-gpu during Live Portrait inference. You can find the CUDA Toolkit corresponding to your `nvidia-smi` version [here](https://developer.nvidia.com/cuda-toolkit-archive).

![image](./nvidia-smi.png)

## Installation logs

The building of the XPose OP dependency adds some logs in the `stable-diffusion-webui/extensions/sd-webui-live-portrait/logs` directory. If an error occurs during the extension installation, you may find useful information in the log files written in this folder.

## Reinstall XPose

To not impact performance when launching WebUI, by default we do not retry XPose installation in case of failure. To force XPose reinstall, you can:
- either restart your SD WebUI and then go to the "Live Portrait" tab and in the "Animals" sub-tab click on the "Reinstall XPose and Restart WebUI" button (only visible if previous installation of XPose has failed),
- or manually delete the `liveportrait/utils/dependencies/XPose/models/UniPose/ops/lib` folder in the extension folder, which is by default `stable-diffusion-webui/extensions/sd-webui-live-portrait` (in that case, folder to remove is `stable-diffusion-webui/extensions/sd-webui-live-portrait/liveportrait/utils/dependencies/XPose/models/UniPose/ops/lib`) and restart your SD WebUI.
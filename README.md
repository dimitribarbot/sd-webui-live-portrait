# [LivePortrait](https://github.com/KwaiVGI/LivePortrait) for Stable Diffusion WebUI

> [!Note]
> Make sure your system has [`FFmpeg`](https://ffmpeg.org/download.html) installed. For details on FFmpeg installation, see [**how to install FFmpeg**](assets/docs/how-to-install-ffmpeg.md).

Pickle files have all been converted to safetensors by Kijai: https://huggingface.co/Kijai/LivePortrait_safetensors/tree/main (thanks to him).  
Model files go here (and are automatically downloaded if the folder is not present): `stable-diffusion-webui/models/liveportrait` (human) and `stable-diffusion-webui/models/liveportrait_animals` (animals).

## Face detectors
For human mode, this extension is using Insightface, which is strictly for NON-COMMERCIAL use.

Insightface models go here (and are automatically downloaded if the folder is not present): `stable-diffusion-webui/models/insightface/models/buffalo_l`.  
If necessary, they can be downloaded from: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip.

For animal mode, this extension is using XPose which is also strictly for NON-COMMERCIAL use and is not compatible with MacOS.

XPose model goes here (and is automatically downloaded if not present): `stable-diffusion-webui/models/liveportrait_animals`.  
If necessary, it can be downloaded from: https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/liveportrait_animals/xpose.pth.

## API
2 routes have been added to the Automatic1111 API:
- `/live-portrait/human`: inference for humans.
- `/live-portrait/animal` : inference for animals (not available for MacOS users).

Parameters are the same as LivePortrait ones (see output of command `python inference.py --help` in LivePortrait repository) except for :
- `source`: it can be either a path to an existing file (as in LivePortrait) or an url or a base64 encoded string. For url without file extension or base64 encoded string, the parameter `source_file_extension` must also be filled with a valid extension corresponding to the given source (e.g. `.jpg`).
- `driving`: it can be either a path to an existing file (as in LivePortrait) or an url or a base64 encoded string. For url without file extension or base64 encoded string, the parameter `driving_file_extension` must also be filled with a valid extension corresponding to the given driving video (e.g. `.mp4`).
- `send_output`: `true` if you want output videos to be sent as base64 encoded strings, `false` otherwise.
- `save_output`: `true` if you want output videos to be saved in `output_dir` (as in LivePortrait), `false` otherwise.  

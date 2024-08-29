## Install FFmpeg

Make sure you have `ffmpeg` and `ffprobe` installed on your system. If you don't have them installed, follow the instructions below.

> [!Note]
> The installation is copied from [SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 🤗

### Conda Users

```bash
conda install ffmpeg
```

### Ubuntu/Debian Users

```bash
sudo apt install ffmpeg
sudo apt install libsox-dev
conda install -c conda-forge 'ffmpeg<7'
```

### Windows Users

Download and place [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe) and [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe) in the folder of your choice. Then add this folder to your PATH environment variable and restart Automatic1111's SD WebUI (follow [this tutorial](https://www.wikihow.com/Change-the-PATH-Environment-Variable-on-Windows) if you don't know how to do that).

### MacOS Users
```bash
brew install ffmpeg
```

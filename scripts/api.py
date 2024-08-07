import base64
import datetime
import os
import requests
import shutil
import sys
import tempfile
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
import gradio as gr
from pydantic import BaseModel
from typing import Any, Literal

from modules.api.api import verify_url
from modules.shared import opts

from scripts.utils import is_valid_cuda_version, isMacOS

from liveportrait.config.argument_config import ArgumentConfig
from liveportrait.config.crop_config import CropConfig
from liveportrait.config.inference_config import InferenceConfig
from liveportrait.live_portrait_pipeline import LivePortraitPipeline
from liveportrait.live_portrait_pipeline_animal import LivePortraitPipelineAnimal
from liveportrait.config.base_config import make_abs_path


temp_dir = make_abs_path('../../tmp')


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_args(args: ArgumentConfig):
    if not args.source:
        raise ValueError("Source info is not optional")
    if not args.driving:
        raise ValueError("Driving info is not optional")


def save_input_to_file(input, filename):
    if os.path.exists(input):
        shutil.copyfile(input, filename)
        return

    if input.startswith("http://") or input.startswith("https://"):
        if not opts.api_enable_requests:
            raise HTTPException(status_code=500, detail="Requests not allowed")

        if opts.api_forbid_local_requests and not verify_url(input):
            raise HTTPException(status_code=500, detail="Request to local resource not allowed")

        headers = {'user-agent': opts.api_useragent} if opts.api_useragent else {}
        try:
            with requests.get(input, timeout=30, headers=headers, stream=True) as r:
                with open(filename, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            return
        except Exception as e:
            raise HTTPException(status_code=500, detail="Invalid input url") from e

    if input.startswith(("data:image/", "data:video/")):
        input = input.split(";")[1].split(",")[1]
    try:
        with open(filename, "wb") as f:
            f.write(base64.b64decode(input))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Invalid encoded input") from e
    

def get_input_extension(input, input_default_extension=None):
    if os.path.exists(input) or (type(input) is str and (input.startswith("http://") or input.startswith("https://"))):
        _, file_extension = os.path.splitext(input)
        if file_extension:
            return file_extension.lower()
    if input_default_extension:
        return input_default_extension.lower()
    raise ValueError("Invalid input extension. You must provide a file name with valid extension or fill in the source_file_extension or driving_file_extension parameter.")
    

def get_output_path(output_dir):
    if os.path.isabs(output_dir):
        return output_dir
    from modules.paths_internal import data_path
    return os.path.join(data_path, output_dir)


def live_portrait_api(_: gr.Blocks, app: FastAPI):
    class LivePortraitRequest(BaseModel):
        source: str = ""  # path to the source portrait (human/animal) or video (human) or base64 encoded one
        source_file_extension: str = ".jpg"  # source file extension if source is a base64 encoded string or url
        driving: str = ""  # path to driving video or template (.pkl format) or base64 encoded one
        driving_file_extension: str = ".mp4"  # driving file extension if driving is a base64 encoded string or url
        output_dir: str = 'outputs/live-portrait/'  # directory to save output video
        send_output: bool = True
        save_output: bool = False

        ########## inference arguments ##########
        flag_use_half_precision: bool = True  # whether to use half precision (FP16). If black boxes appear, it might be due to GPU incompatibility; set to False.
        flag_crop_driving_video: bool = False  # whether to crop the driving video, if the given driving info is a video
        device_id: int = 0  # gpu device id
        flag_force_cpu: bool = False  # force cpu inference, WIP!
        flag_normalize_lip: bool = True  # whether to let the lip to close state before animation, only take effect when flag_eye_retargeting and flag_lip_retargeting is False
        flag_source_video_eye_retargeting: bool = False  # when the input is a source video, whether to let the eye-open scalar of each frame to be the same as the first source frame before the animation, only take effect when flag_eye_retargeting and flag_lip_retargeting is False, may cause the inter-frame jittering
        flag_video_editing_head_rotation: bool = False  # when the input is a source video, whether to inherit the relative head rotation from the driving video
        flag_eye_retargeting: bool = False  # not recommend to be True, WIP; whether to transfer the eyes-open ratio of each driving frame to the source image or the corresponding source frame
        flag_lip_retargeting: bool = False  # not recommend to be True, WIP; whether to transfer the lip-open ratio of each driving frame to the source image or the corresponding source frame
        flag_stitching: bool = True  # recommend to True if head movement is small, False if head movement is large or the source image is an animal
        flag_relative_motion: bool = True # whether to use relative motion
        flag_pasteback: bool = True  # whether to paste-back/stitch the animated face cropping from the face-cropping space to the original image space
        flag_do_crop: bool = True  # whether to crop the source portrait or video to the face-cropping space
        driving_option: Literal["expression-friendly", "pose-friendly"] = "expression-friendly" # "expression-friendly" or "pose-friendly"; "expression-friendly" would adapt the driving motion with the global multiplier, and could be used when the source is a human image
        driving_multiplier: float = 1.0 # be used only when driving_option is "expression-friendly"
        driving_smooth_observation_variance: float = 3e-7  # smooth strength scalar for the animated video when the input is a source video, the larger the number, the smoother the animated video; too much smoothness would result in loss of motion accuracy
        audio_priority: Literal['source', 'driving'] = 'driving'  # whether to use the audio from source or driving video
        ########## source crop arguments ##########
        det_thresh: float = 0.15 # detection threshold
        scale: float = 2.3  # the ratio of face area is smaller if scale is larger
        vx_ratio: float = 0  # the ratio to move the face to left or right in cropping space
        vy_ratio: float = -0.125  # the ratio to move the face to up or down in cropping space
        flag_do_rot: bool = True  # whether to conduct the rotation when flag_do_crop is True
        source_max_dim: int = 1280 # the max dim of height and width of source image or video, you can change it to a larger number, e.g., 1920
        source_division: int = 2 # make sure the height and width of source image or video can be divided by this number

        ########## driving crop arguments ##########
        scale_crop_driving_video: float = 2.2  # scale factor for cropping driving video
        vx_ratio_crop_driving_video: float = 0.  # adjust y offset
        vy_ratio_crop_driving_video: float = -0.1  # adjust x offset


    @app.post("/live-portrait/human")
    async def execute_human(payload: LivePortraitRequest = Body(...)) -> Any:
        print("Live Portrait API /live-portrait/human received request")
        os.makedirs(temp_dir, exist_ok=True)

        source_extension = get_input_extension(payload.source, payload.source_file_extension)
        driving_extension = get_input_extension(payload.driving, payload.driving_file_extension)

        argument_cfg = partial_fields(ArgumentConfig, payload.__dict__)

        fast_check_args(argument_cfg)

        with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdirname:
            temp_output_dir  = os.path.join(tmpdirname, "output")
            os.makedirs(temp_output_dir, exist_ok=True)

            temp_source_file = tempfile.NamedTemporaryFile(dir=tmpdirname, suffix=source_extension)
            temp_source_file.close()
            save_input_to_file(argument_cfg.source, temp_source_file.name)
            argument_cfg.source = temp_source_file.name

            temp_driving_file = tempfile.NamedTemporaryFile(dir=tmpdirname, suffix=driving_extension)
            temp_driving_file.close()
            save_input_to_file(argument_cfg.driving, temp_driving_file.name)
            argument_cfg.driving = temp_driving_file.name

            argument_cfg.output_dir = temp_output_dir

            live_portrait_pipeline = LivePortraitPipeline(
                inference_cfg=InferenceConfig(),
                crop_cfg=CropConfig()
            )

            wfp, wfp_concat = live_portrait_pipeline.execute(argument_cfg)

            if payload.save_output:
                ouput_path = get_output_path(payload.output_dir)
                timestamped_output_path = os.path.join(ouput_path, f"{datetime.date.today()}")
                shutil.copytree(temp_output_dir, timestamped_output_path, dirs_exist_ok=True)

            if payload.send_output:
                with open(wfp, 'rb') as wfp_f:
                    animated_video = (base64.b64encode(wfp_f.read())).decode()
                with open(wfp_concat, 'rb') as wfp_concat_f:
                    animated_video_with_concat = (base64.b64encode(wfp_concat_f.read())).decode()
            else:
                animated_video = None
                animated_video_with_concat = None

            print("Live Portrait API /live-portrait/human finished")

            return {"animated_video": animated_video, "animated_video_with_concat": animated_video_with_concat }


    @app.post("/live-portrait/animal")
    async def execute_animal(payload: LivePortraitRequest = Body(...)) -> Any:
        print("Live Portrait API /live-portrait/animal received request")
        if isMacOS():
            raise OSError("XPose model, necessary to generate animal videos, is incompatible with MacOS systems.")
        if not is_valid_cuda_version():
            raise SystemError("XPose model, necessary to generate animal videos, is incompatible with pytorch version 2.1.x.")
        
        os.makedirs(temp_dir, exist_ok=True)

        source_extension = get_input_extension(payload.source, payload.source_file_extension)
        driving_extension = get_input_extension(payload.driving, payload.driving_file_extension)

        argument_cfg = partial_fields(ArgumentConfig, payload.__dict__)

        # argument_cfg.driving_multiplier=1.75
        # argument_cfg.flag_stitching=False

        fast_check_args(argument_cfg)

        with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdirname:
            temp_output_dir  = os.path.join(tmpdirname, "output")
            os.makedirs(temp_output_dir, exist_ok=True)

            temp_source_file = tempfile.NamedTemporaryFile(dir=tmpdirname, suffix=source_extension)
            temp_source_file.close()
            save_input_to_file(argument_cfg.source, temp_source_file.name)
            argument_cfg.source = temp_source_file.name

            temp_driving_file = tempfile.NamedTemporaryFile(dir=tmpdirname, suffix=driving_extension)
            temp_driving_file.close()
            save_input_to_file(argument_cfg.driving, temp_driving_file.name)
            argument_cfg.driving = temp_driving_file.name

            argument_cfg.output_dir = temp_output_dir

            live_portrait_pipeline_animal = LivePortraitPipelineAnimal(
                inference_cfg=InferenceConfig(),
                crop_cfg=CropConfig()
            )

            wfp, wfp_concat, wfp_gif = live_portrait_pipeline_animal.execute(argument_cfg)

            if payload.save_output:
                ouput_path = get_output_path(payload.output_dir)
                timestamped_output_path = os.path.join(ouput_path, f"{datetime.date.today()}")
                shutil.copytree(temp_output_dir, timestamped_output_path, dirs_exist_ok=True)

            if payload.send_output:
                with open(wfp, 'rb') as wfp_f:
                    animated_video = (base64.b64encode(wfp_f.read())).decode()
                with open(wfp_concat, 'rb') as wfp_concat_f:
                    animated_video_with_concat = (base64.b64encode(wfp_concat_f.read())).decode()
                with open(wfp_gif, 'rb') as wfp_gif_f:
                    animated_gif = (base64.b64encode(wfp_gif_f.read())).decode()
            else:
                animated_video = None
                animated_video_with_concat = None
                animated_gif = None
        
            print("Live Portrait API /live-portrait/animal finished")

            return {"animated_video": animated_video, "animated_video_with_concat": animated_video_with_concat, "animated_gif": animated_gif }


try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(live_portrait_api)
except:
    print("Live Portrait API failed to initialize")
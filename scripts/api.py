import base64
import datetime
import imageio.v3 as iio
import os
import requests
import shutil
import tempfile
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
import gradio as gr
from pydantic import BaseModel
from typing import Any, cast, Dict, List, Literal
import cv2

from liveportrait.gradio_pipeline import GradioPipeline
from modules.api.api import verify_url
from modules.devices import torch_gc
from modules.shared import opts

from liveportrait.config.argument_config import ArgumentConfig
from liveportrait.config.base_config import make_abs_path
from liveportrait.config.crop_config import CropConfig
from liveportrait.config.inference_config import InferenceConfig
from liveportrait.live_portrait_pipeline import LivePortraitPipeline
from liveportrait.live_portrait_pipeline_animal import LivePortraitPipelineAnimal
from liveportrait.utils.helper import basename

from internal_liveportrait.utils import \
    download_insightface_models, download_liveportrait_animals_models, download_liveportrait_models, download_liveportrait_landmark_model, \
    IS_MACOS, has_xpose_lib


temp_dir = make_abs_path('../../tmp')

live_portrait_pipeline: LivePortraitPipeline | None = None
live_portrait_pipeline_animal: LivePortraitPipelineAnimal | None = None
retargeting_pipeline: GradioPipeline | None = None

def clear_model_cache():
    global live_portrait_pipeline, live_portrait_pipeline_animal, retargeting_pipeline
    live_portrait_pipeline = None
    live_portrait_pipeline_animal = None
    retargeting_pipeline = None
    torch_gc()

def init_live_portrait_pipeline(inference_cfg: InferenceConfig, crop_cfg: CropConfig, use_model_cache: bool):
    global live_portrait_pipeline, live_portrait_pipeline_animal, retargeting_pipeline
    if not use_model_cache:
        clear_model_cache()
        return LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
        )
    if not live_portrait_pipeline or live_portrait_pipeline.cropper.crop_cfg.model != crop_cfg.model or \
        live_portrait_pipeline.cropper.crop_cfg.flag_force_cpu != crop_cfg.flag_force_cpu or \
        live_portrait_pipeline.cropper.crop_cfg.device_id != crop_cfg.device_id or \
        (crop_cfg.model == "facealignment" and (live_portrait_pipeline.cropper.crop_cfg.face_alignment_detector != crop_cfg.face_alignment_detector \
                                                or live_portrait_pipeline.cropper.crop_cfg.face_alignment_detector_device != crop_cfg.face_alignment_detector_device
                                                or live_portrait_pipeline.cropper.crop_cfg.face_alignment_detector_dtype != crop_cfg.face_alignment_detector_dtype)):
        clear_model_cache()
        live_portrait_pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
        )
    else:
        live_portrait_pipeline.cropper.update_config(crop_cfg.__dict__)
        live_portrait_pipeline.live_portrait_wrapper.update_config(inference_cfg.__dict__)
    return live_portrait_pipeline


def init_live_portrait_animal_pipeline(inference_cfg: InferenceConfig, crop_cfg: CropConfig, use_model_cache: bool):
    global live_portrait_pipeline, live_portrait_pipeline_animal, retargeting_pipeline
    if not use_model_cache:
        clear_model_cache()
        return LivePortraitPipelineAnimal(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
        )
    if not live_portrait_pipeline_animal or live_portrait_pipeline_animal.cropper.crop_cfg.model != crop_cfg.model or \
        live_portrait_pipeline_animal.cropper.crop_cfg.flag_force_cpu != crop_cfg.flag_force_cpu or \
        live_portrait_pipeline_animal.cropper.crop_cfg.device_id != crop_cfg.device_id or \
        (crop_cfg.model == "facealignment" and (live_portrait_pipeline_animal.cropper.crop_cfg.face_alignment_detector != crop_cfg.face_alignment_detector \
                                                or live_portrait_pipeline_animal.cropper.crop_cfg.face_alignment_detector_device != crop_cfg.face_alignment_detector_device
                                                or live_portrait_pipeline_animal.cropper.crop_cfg.face_alignment_detector_dtype != crop_cfg.face_alignment_detector_dtype)):
        clear_model_cache()
        live_portrait_pipeline_animal = LivePortraitPipelineAnimal(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
        )
    else:
        live_portrait_pipeline_animal.cropper.update_config(crop_cfg.__dict__)
        live_portrait_pipeline_animal.live_portrait_wrapper_animal.update_config(inference_cfg.__dict__)
    return live_portrait_pipeline_animal


def init_retargeting_pipeline(inference_cfg: InferenceConfig, crop_cfg: CropConfig, argument_cfg: ArgumentConfig, use_model_cache: bool):
    global live_portrait_pipeline, live_portrait_pipeline_animal, retargeting_pipeline
    if not use_model_cache:
        clear_model_cache()
        return GradioPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg,
            args=argument_cfg
        )
    if not retargeting_pipeline or retargeting_pipeline.cropper.crop_cfg.model != crop_cfg.model or \
        retargeting_pipeline.cropper.crop_cfg.flag_force_cpu != crop_cfg.flag_force_cpu or \
        retargeting_pipeline.cropper.crop_cfg.device_id != crop_cfg.device_id or \
        (crop_cfg.model == "facealignment" and (retargeting_pipeline.cropper.crop_cfg.face_alignment_detector != crop_cfg.face_alignment_detector \
                                                or retargeting_pipeline.cropper.crop_cfg.face_alignment_detector_device != crop_cfg.face_alignment_detector_device
                                                or retargeting_pipeline.cropper.crop_cfg.face_alignment_detector_dtype != crop_cfg.face_alignment_detector_dtype)):
        clear_model_cache()
        retargeting_pipeline = GradioPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg,
            args=argument_cfg
        )
    else:
        retargeting_pipeline.cropper.update_config(crop_cfg.__dict__)
        retargeting_pipeline.live_portrait_wrapper.update_config(inference_cfg.__dict__)
    return retargeting_pipeline


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def save_input_to_temp_file(input: str, input_extension: str, tmpdirname: str):
    temp_source_file = tempfile.NamedTemporaryFile(dir=tmpdirname, suffix=input_extension)
    temp_source_file.close()
    temp_source_file_name = temp_source_file.name

    if os.path.exists(input):
        shutil.copyfile(input, temp_source_file_name)
        return temp_source_file_name

    if input.startswith("http://") or input.startswith("https://"):
        if not opts.api_enable_requests:
            raise HTTPException(status_code=500, detail="Requests not allowed")

        if opts.api_forbid_local_requests and not verify_url(input):
            raise HTTPException(status_code=500, detail="Request to local resource not allowed")

        headers = {'user-agent': opts.api_useragent} if opts.api_useragent else {}
        try:
            with requests.get(input, timeout=30, headers=headers, stream=True) as r:
                with open(temp_source_file_name, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            return temp_source_file_name
        except Exception as e:
            raise HTTPException(status_code=500, detail="Invalid input url") from e

    if input.startswith(("data:image/", "data:video/")):
        input = input.split(";")[1].split(",")[1]
    try:
        with open(temp_source_file_name, "wb") as f:
            f.write(base64.b64decode(input))
        return temp_source_file_name
    except Exception as e:
        raise HTTPException(status_code=500, detail="Invalid encoded input") from e
    

def is_file(input):
    return os.path.exists(input) or (type(input) is str and (input.startswith("http://") or input.startswith("https://")))
    

def get_input_extension(input, input_default_extension=None):
    if is_file(input):
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


def initialize_crop_model(
        crop_cfg: CropConfig,
        human_face_detector: Literal[None, 'insightface', 'mediapipe', 'facealignment'] = None,
        face_alignment_detector: Literal[None, 'blazeface', 'blazeface_back_camera', 'retinaface', 'sfd'] = None,
        face_alignment_detector_device: Literal[None, 'cuda', 'cpu', 'mps'] = None,
        face_alignment_detector_dtype: Literal[None, 'fp16', 'bf16', 'fp32'] = None):
    
    default_crop_model = cast(
        Literal['insightface', 'mediapipe', 'facealignment'],
        cast(str, opts.data.get("live_portrait_human_face_detector", 'insightface')).lower()
    )

    default_face_alignment_detector = cast(
        Literal['blazeface', 'blazeface_back_camera', 'retinaface', 'sfd'],
        cast(str, opts.data.get("live_portrait_face_alignment_detector", 'blazeface_back_camera')).lower().replace(' ', '_')
    )

    default_face_alignment_detector_device = cast(
        Literal['cuda', 'cpu', 'mps'],
        cast(str, opts.data.get("live_portrait_face_alignment_detector_device", 'cuda')).lower()
    )

    default_face_alignment_detector_dtype = cast(
        Literal['fp16', 'bf16', 'fp32'],
        cast(str, opts.data.get("live_portrait_face_alignment_detector_dtype", 'fp16')).lower()
    )

    crop_cfg.model = human_face_detector if human_face_detector else default_crop_model
    crop_cfg.face_alignment_detector = face_alignment_detector if face_alignment_detector else default_face_alignment_detector
    crop_cfg.face_alignment_detector_device = face_alignment_detector_device if face_alignment_detector_device else default_face_alignment_detector_device
    crop_cfg.face_alignment_detector_dtype = face_alignment_detector_dtype if face_alignment_detector_dtype else default_face_alignment_detector_dtype
    
    return crop_cfg


def rename_output_videos(wfp: str, wfp_concat: str, temp_output_dir: str, new_names_to_old_names: Dict[str, str]):
    new_name_wfp = os.path.basename(wfp)
    new_name_wfp_concat = os.path.basename(wfp_concat)
    for new_name, old_name in new_names_to_old_names.items():
        new_name_wfp = new_name_wfp.replace(new_name, old_name)
        new_name_wfp_concat = new_name_wfp_concat.replace(new_name, old_name)
    if new_name_wfp != os.path.basename(wfp):
        new_wfp = os.path.join(temp_output_dir, new_name_wfp)
        os.rename(wfp, new_wfp)
        wfp = new_wfp
    if new_name_wfp_concat != os.path.basename(wfp_concat):
        new_wfp_concat = os.path.join(temp_output_dir, new_name_wfp_concat)
        os.rename(wfp_concat, new_wfp_concat)
        wfp_concat = new_wfp_concat
    return wfp, wfp_concat


def rename_output_videos_and_gif(wfp: str, wfp_concat: str, wfp_gif: str, temp_output_dir: str, new_names_to_old_names: Dict[str, str]):
    wfp, wfp_concat = rename_output_videos(wfp, wfp_concat, temp_output_dir, new_names_to_old_names)
    new_name_wfp_gif = os.path.basename(wfp_gif)
    for new_name, old_name in new_names_to_old_names.items():
        new_name_wfp_gif = new_name_wfp_gif.replace(new_name, old_name)
    if new_name_wfp_gif != os.path.basename(wfp_gif):
        new_wfp_gif = os.path.join(temp_output_dir, new_name_wfp_gif)
        os.rename(wfp_gif, new_wfp_gif)
        wfp_gif = new_wfp_gif
    return wfp, wfp_concat, wfp_gif


def save_files_output(output_dir: str, temp_output_dir: str):
    ouput_path = get_output_path(output_dir)
    timestamped_output_path = os.path.join(ouput_path, f"{datetime.date.today()}")
    shutil.copytree(temp_output_dir, timestamped_output_path, dirs_exist_ok=True)


def save_video_frames_to_ouput(
        wfp: str,
        wfp_concat: str,
        frame_indices: List[int],
        output_dir: str,
        temp_output_dir: str,
        source_file_extension: str,
        save_output: bool,
        send_output: bool):
    
    animated_images = []
    animated_images_with_concat = []

    with iio.imopen(wfp, "r") as wfp_f:
        for frame_index in frame_indices:
            if frame_index > 0:
                frame = wfp_f.read(index=frame_index-1)
                if save_output:
                    wfp_path = os.path.join(os.path.dirname(wfp), f"{basename(wfp)}_{frame_index}{source_file_extension}")
                    iio.imwrite(wfp_path, frame, extension=source_file_extension)
                if send_output:
                    animated_image = iio.imwrite("<bytes>", frame, extension=source_file_extension)
                    animated_image = (base64.b64encode(animated_image)).decode()
                    animated_images.append(animated_image)
    with iio.imopen(wfp_concat, "r") as wfp_concat_f:
        for frame_index in frame_indices:
            if frame_index > 0:
                frame = wfp_concat_f.read(index=frame_index-1)
                if save_output:
                    wfp_concat_path = os.path.join(os.path.dirname(wfp_concat), f"{basename(wfp_concat)}_{frame_index}{source_file_extension}")
                    iio.imwrite(wfp_concat_path, frame, extension=source_file_extension)
                if send_output:
                    animated_image_with_concat = iio.imwrite("<bytes>", frame, extension=source_file_extension)
                    animated_image_with_concat = (base64.b64encode(animated_image_with_concat)).decode()
                    animated_images_with_concat.append(animated_image_with_concat)

    if save_output:
        os.remove(wfp)
        os.remove(wfp_concat)
        save_files_output(output_dir, temp_output_dir)

    return animated_images, animated_images_with_concat


def save_videos_to_ouput(wfp: str, wfp_concat: str, output_dir: str, temp_output_dir: str, save_output: bool, send_output: bool):
    if save_output:
        save_files_output(output_dir, temp_output_dir)

    if send_output:
        with open(wfp, 'rb') as wfp_f:
            animated_video = (base64.b64encode(wfp_f.read())).decode()
        with open(wfp_concat, 'rb') as wfp_concat_f:
            animated_video_with_concat = (base64.b64encode(wfp_concat_f.read())).decode()
    else:
        animated_video = None
        animated_video_with_concat = None
    
    return animated_video, animated_video_with_concat


def save_videos_and_gif_to_ouput(wfp: str, wfp_concat: str, wfp_gif: str, output_dir: str, temp_output_dir: str, save_output: bool, send_output: bool):
    animated_video, animated_video_with_concat = save_videos_to_ouput(wfp, wfp_concat, output_dir, temp_output_dir, save_output, send_output)

    if send_output:
        with open(wfp_gif, 'rb') as wfp_gif_f:
            animated_gif = (base64.b64encode(wfp_gif_f.read())).decode()
    else:
        animated_gif = None
    
    return animated_video, animated_video_with_concat, animated_gif


def live_portrait_api(_: gr.Blocks, app: FastAPI):
    
    class LivePortraitRequest(BaseModel):
        source: str = ""  # path to the source portrait (human/animal) or video (human) or base64 encoded one
        source_file_extension: str = ".jpg"  # source file extension if source is a base64 encoded string or url
        driving: str = ""  # path to driving image, video or template (.pkl format) or base64 encoded one
        driving_file_extension: str = ".mp4"  # driving file extension if driving is a base64 encoded string or url
        output_dir: str = 'outputs/live-portrait/'  # directory to save output video
        output_mode: Literal["video", "images"] = "video"  # whether to generate an output video or multiple images at given frame indices (see frame_indices)
        frame_indices: List[int] = []  # frame indices where image snapshots should be generated (to be used with the 'images' output mode)
        send_output: bool = True
        save_output: bool = False
        use_model_cache: bool = True

        ########## inference arguments ##########
        flag_use_half_precision: bool = True  # whether to use half precision (FP16). If black boxes appear, it might be due to GPU incompatibility; set to False.
        flag_crop_driving_video: bool = False  # whether to crop the driving video, if the given driving info is a video
        device_id: int = 0  # gpu device id
        flag_force_cpu: bool = False  # force cpu inference, WIP!
        flag_normalize_lip: bool = False  # whether to let the lip to close state before animation, only take effect when flag_eye_retargeting and flag_lip_retargeting is False
        flag_source_video_eye_retargeting: bool = False  # when the input is a source video, whether to let the eye-open scalar of each frame to be the same as the first source frame before the animation, only take effect when flag_eye_retargeting and flag_lip_retargeting is False, may cause the inter-frame jittering
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
        animation_region: Literal["exp", "pose", "lip", "eyes", "all"] = "all" # the region where the animation was performed, "exp" means the expression, "pose" means the head pose, "all" means all regions
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
        human_face_detector: Literal[None, 'insightface', 'mediapipe', 'facealignment'] = None # face detector to use for human inference ('insightface' by default)
        face_alignment_detector: Literal[None, 'blazeface', 'blazeface_back_camera', 'retinaface', 'sfd'] = None
        face_alignment_detector_device: Literal[None, 'cuda', 'cpu', 'mps'] = None
        face_alignment_detector_dtype: Literal[None, 'fp16', 'bf16', 'fp32'] = None

        ########## face index ##########
        source_face_index: int = 0  # source image or video face index
        driving_face_index: int = 0  # driving video face index


    def fast_check_inference_args(payload: LivePortraitRequest):
        if not payload.source:
            raise ValueError("Source info is not optional")
        if not payload.driving:
            raise ValueError("Driving info is not optional")
        if payload.output_mode == "images" and (payload.frame_indices is None or len(payload.frame_indices) == 0):
            raise ValueError("Frame indices are not optional when using 'images' output mode")


    @app.post("/live-portrait/human")
    async def execute_human(payload: LivePortraitRequest = Body(...)) -> Any:
        print("Live Portrait API /live-portrait/human received request")

        fast_check_inference_args(payload)

        argument_cfg = partial_fields(ArgumentConfig, payload.__dict__)
        inference_cfg = partial_fields(InferenceConfig, payload.__dict__)
        crop_cfg = partial_fields(CropConfig, payload.__dict__)

        os.makedirs(temp_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdirname:
            temp_output_dir = os.path.join(tmpdirname, "output")
            os.makedirs(temp_output_dir, exist_ok=True)

            source_file_extension = get_input_extension(payload.source, payload.source_file_extension)
            driving_file_extension = get_input_extension(payload.driving, payload.driving_file_extension)
            
            argument_cfg.source = save_input_to_temp_file(payload.source, source_file_extension, tmpdirname)
            argument_cfg.driving = save_input_to_temp_file(payload.driving, driving_file_extension, tmpdirname)

            source_tmp_name = basename(argument_cfg.source)
            driving_tmp_name = basename(argument_cfg.driving)
            new_names_to_old_names = {
                source_tmp_name: basename(payload.source) if is_file(payload.source) else "source",
                driving_tmp_name: basename(payload.driving) if is_file(payload.driving) else "driving"
            }

            argument_cfg.output_dir = temp_output_dir

            initialize_crop_model(
                crop_cfg,
                payload.human_face_detector,
                payload.face_alignment_detector,
                payload.face_alignment_detector_device,
                payload.face_alignment_detector_dtype
            )
            
            download_liveportrait_models()
            if crop_cfg.model == "insightface":
                download_insightface_models()

            live_portrait_pipeline = init_live_portrait_pipeline(
                inference_cfg,
                crop_cfg,
                payload.use_model_cache
            )

            wfp, wfp_concat = live_portrait_pipeline.execute(argument_cfg)
            wfp, wfp_concat = rename_output_videos(wfp, wfp_concat, temp_output_dir, new_names_to_old_names)

            if payload.output_mode == "images":
                animated_images, animated_images_with_concat = save_video_frames_to_ouput(
                    wfp,
                    wfp_concat,
                    payload.frame_indices,
                    payload.output_dir,
                    temp_output_dir,
                    source_file_extension,
                    payload.save_output,
                    payload.send_output
                )
                print("Live Portrait API /live-portrait/human finished")
                return {"animated_images": animated_images, "animated_images_with_concat": animated_images_with_concat }  
            
            animated_video, animated_video_with_concat = save_videos_to_ouput(
                wfp,
                wfp_concat,
                payload.output_dir,
                temp_output_dir,
                payload.save_output,
                payload.send_output
            )
            print("Live Portrait API /live-portrait/human finished")
            return {"animated_video": animated_video, "animated_video_with_concat": animated_video_with_concat }
        

    class LivePortraitImageRetargetingOptions(BaseModel):  
        eye_ratio: float = 0  # target eyes-open ratio (0 -> 0.8)
        lip_ratio: float = 0  # target lip-open ratio (0 -> 0.8)
        head_pitch_variation: float = 0  # relative pitch (-15 -> 15)
        head_yaw_variation: float = 0  # relative yaw (-25 -> 25)
        head_roll_variation: float = 0  # relative roll (-15 -> 15)
        mov_x: float = 0  # x-axis movement (-0.19 -> 0.19)
        mov_y: float = 0  # y-axis movement (-0.19 -> 0.19)
        mov_z: float = 1  # z-axis movement (0.9 -> 1.2)
        lip_variation_pouting: float = 0  # (-0.09 -> 0.09)
        lip_variation_pursing: float = 0  # (-20 -> 15)
        lip_variation_grin: float = 0  # (0 -> 15)
        lip_variation_opening: float = 0  # lip close <-> open (-90 -> 120)
        smile: float = 0  # (-0.3 -> 1.3)
        wink: float = 0  # (0 -> 39)
        eyebrow: float = 0  # (-30 -> 30)
        eyeball_direction_x: float = 0  # eye gaze (horizontal) (-30 -> 30)
        eyeball_direction_y: float = 0  # eye gaze (vertical) (-63 -> 63)


    class LivePortraitCommonRetargetingRequest(BaseModel):
        source: str = ""  # path to the source portrait or base64 encoded one
        use_model_cache: bool = True

        ########## source crop arguments ##########
        device_id: int = 0  # gpu device id
        flag_force_cpu: bool = False  # force cpu inference, WIP!
        det_thresh: float = 0.15 # detection threshold
        vx_ratio: float = 0  # the ratio to move the face to left or right in cropping space
        vy_ratio: float = -0.125  # the ratio to move the face to up or down in cropping space
        flag_do_rot: bool = True  # whether to conduct the rotation when flag_do_crop is True
        human_face_detector: Literal[None, 'insightface', 'mediapipe', 'facealignment'] = None # face detector to use for human inference ('insightface' by default)
        face_alignment_detector: Literal[None, 'blazeface', 'blazeface_back_camera', 'retinaface', 'sfd'] = None
        face_alignment_detector_device: Literal[None, 'cuda', 'cpu', 'mps'] = None
        face_alignment_detector_dtype: Literal[None, 'fp16', 'bf16', 'fp32'] = None
        source_face_index: int = 0  # source image or video face index


    class LivePortraitImageRetargetingRequest(LivePortraitCommonRetargetingRequest):
        source_file_extension: str = ".jpg"  # source file extension if source is a base64 encoded string or url
        output_dir: str = 'outputs/live-portrait/'  # directory to save output video
        send_output: bool = True
        save_output: bool = False

        ########## retargeting arguments ##########
        retargeting_options: List[LivePortraitImageRetargetingOptions] = [
            LivePortraitImageRetargetingOptions()
        ]
        retargeting_source_scale: float = 2.5  # the ratio of face area is smaller if scale is larger
        flag_stitching_retargeting_input: bool = True  # To apply stitching or not
        flag_do_crop_input_retargeting_image: bool = True  # whether to crop the source portrait to the face-cropping space


    def fast_check_retargeting_args(payload: LivePortraitCommonRetargetingRequest):
        if not payload.source:
            raise ValueError("Source info is not optional")


    @app.post("/live-portrait/human/retargeting/image")
    async def execute_image_retargeting(payload: LivePortraitImageRetargetingRequest = Body(...)) -> Any:
        print("Live Portrait API /live-portrait/human/retargeting/image received request")

        fast_check_retargeting_args(payload)

        argument_cfg = partial_fields(ArgumentConfig, payload.__dict__)
        inference_cfg = partial_fields(InferenceConfig, payload.__dict__)
        crop_cfg = partial_fields(CropConfig, payload.__dict__)

        os.makedirs(temp_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdirname:
            temp_output_dir  = os.path.join(tmpdirname, "output")
            os.makedirs(temp_output_dir, exist_ok=True)

            source_file_extension = get_input_extension(payload.source, payload.source_file_extension)

            argument_cfg.source = save_input_to_temp_file(payload.source, source_file_extension, tmpdirname)

            argument_cfg.output_dir = temp_output_dir

            initialize_crop_model(
                crop_cfg,
                payload.human_face_detector,
                payload.face_alignment_detector,
                payload.face_alignment_detector_device,
                payload.face_alignment_detector_dtype
            )
            
            download_liveportrait_models()
            if crop_cfg.model == "insightface":
                download_insightface_models()

            retargeting_pipeline = init_retargeting_pipeline(
                inference_cfg,
                crop_cfg,
                argument_cfg,
                payload.use_model_cache
            )

            retargeting_images = []
            retargeting_images_cropped = []
            for option_index in range(len(payload.retargeting_options)):
                retargeting_option = payload.retargeting_options[option_index]
                out, out_to_ori_blend = retargeting_pipeline.execute_image_retargeting(
                    retargeting_option.eye_ratio,
                    retargeting_option.lip_ratio,
                    retargeting_option.head_pitch_variation,
                    retargeting_option.head_yaw_variation,
                    retargeting_option.head_roll_variation,
                    retargeting_option.mov_x,
                    retargeting_option.mov_y,
                    retargeting_option.mov_z,
                    retargeting_option.lip_variation_pouting,
                    retargeting_option.lip_variation_pursing,
                    retargeting_option.lip_variation_grin,
                    retargeting_option.lip_variation_opening,
                    retargeting_option.smile,
                    retargeting_option.wink,
                    retargeting_option.eyebrow,
                    retargeting_option.eyeball_direction_x,
                    retargeting_option.eyeball_direction_y,
                    argument_cfg.source,
                    payload.source_face_index,
                    payload.retargeting_source_scale,
                    payload.flag_stitching_retargeting_input,
                    payload.flag_do_crop_input_retargeting_image
                )

                suffix = "" if len(payload.retargeting_options) == 1 else f"_{option_index}"

                source_file_name = basename(payload.source) if is_file(payload.source) else "source"

                wfp = os.path.join(temp_output_dir, f'{source_file_name}_retargeting{suffix}{payload.source_file_extension}')
                wfp_concat = os.path.join(temp_output_dir, f'{source_file_name}_retargeting_cropped{suffix}{payload.source_file_extension}')
                cv2.imwrite(wfp, cv2.cvtColor(out_to_ori_blend, cv2.COLOR_BGR2RGB))
                cv2.imwrite(wfp_concat, cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

                if payload.send_output:
                    with open(wfp, 'rb') as wfp_f:
                        retargeting_image = (base64.b64encode(wfp_f.read())).decode()
                    with open(wfp_concat, 'rb') as wfp_concat_f:
                        retargeting_image_cropped = (base64.b64encode(wfp_concat_f.read())).decode()
                    retargeting_images.append(retargeting_image)
                    retargeting_images_cropped.append(retargeting_image_cropped)

            if payload.save_output:
                save_files_output(payload.output_dir, temp_output_dir)

            print("Live Portrait API /live-portrait/human/retargeting/image finished")

            return {"retargeting_images": retargeting_images, "retargeting_images_cropped": retargeting_images_cropped }


    class LivePortraitImageRetargetingInitRequest(LivePortraitCommonRetargetingRequest):
        source_file_extension: str = ".jpg"  # source file extension if source is a base64 encoded string or url

        ########## retargeting arguments ##########
        eye_ratio: float = 0  # target eyes-open ratio (0 -> 0.8)
        lip_ratio: float = 0  # target lip-open ratio (0 -> 0.8)
        retargeting_source_scale: float = 2.5  # the ratio of face area is smaller if scale is larger
        

    @app.post("/live-portrait/human/retargeting/image/init")
    async def init_image_retargeting(payload: LivePortraitImageRetargetingInitRequest = Body(...)) -> Any:
        print("Live Portrait API /live-portrait/human/retargeting/image/init received request")

        fast_check_retargeting_args(payload)
        
        argument_cfg = partial_fields(ArgumentConfig, payload.__dict__)
        inference_cfg = partial_fields(InferenceConfig, payload.__dict__)
        crop_cfg = partial_fields(CropConfig, payload.__dict__)

        os.makedirs(temp_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdirname:
            source_file_extension = get_input_extension(payload.source, payload.source_file_extension)

            argument_cfg.source = save_input_to_temp_file(payload.source, source_file_extension, tmpdirname)

            initialize_crop_model(
                crop_cfg,
                payload.human_face_detector,
                payload.face_alignment_detector,
                payload.face_alignment_detector_device,
                payload.face_alignment_detector_dtype
            )
            
            download_liveportrait_models()
            if crop_cfg.model == "insightface":
                download_insightface_models()

            retargeting_pipeline = init_retargeting_pipeline(
                inference_cfg,
                crop_cfg,
                argument_cfg,
                payload.use_model_cache
            )

            source_eye_ratio, source_lip_ratio = retargeting_pipeline.init_retargeting_image(
                payload.source_face_index,
                payload.retargeting_source_scale,
                payload.eye_ratio,
                payload.lip_ratio,
                argument_cfg.source
            )

            print("Live Portrait API /live-portrait/human/retargeting/image/init finished")

            return {"source_eye_ratio": source_eye_ratio, "source_lip_ratio": source_lip_ratio }

    
    class LivePortraitVideoRetargetingRequest(LivePortraitCommonRetargetingRequest):
        source_file_extension: str = ".mp4"  # source file extension if source is a base64 encoded string or url
        output_dir: str = 'outputs/live-portrait/'  # directory to save output video
        send_output: bool = True
        save_output: bool = False

        ########## retargeting arguments ##########
        lip_ratio: float = 0  # target lip-open ratio (0 -> 0.8)
        retargeting_source_scale: float = 2.3  # the ratio of face area is smaller if scale is larger
        driving_smooth_observation_variance_retargeting: float = 3e-6  # motion smooth strength
        video_retargeting_silence: bool = False  # keeping the lip silent
        flag_do_crop_input_retargeting_video: bool = True  # whether to crop the source video to the face-cropping space


    @app.post("/live-portrait/human/retargeting/video")
    async def execute_video_retargeting(payload: LivePortraitVideoRetargetingRequest = Body(...)) -> Any:
        print("Live Portrait API /live-portrait/human/retargeting/video received request")

        fast_check_retargeting_args(payload)
        
        argument_cfg = partial_fields(ArgumentConfig, payload.__dict__)
        inference_cfg = partial_fields(InferenceConfig, payload.__dict__)
        crop_cfg = partial_fields(CropConfig, payload.__dict__)

        os.makedirs(temp_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdirname:
            temp_output_dir  = os.path.join(tmpdirname, "output")
            os.makedirs(temp_output_dir, exist_ok=True)

            source_file_extension = get_input_extension(payload.source, payload.source_file_extension)

            argument_cfg.source = save_input_to_temp_file(payload.source, source_file_extension, tmpdirname)

            source_tmp_name = basename(argument_cfg.source)
            new_names_to_old_names = {
                source_tmp_name: basename(payload.source) if is_file(payload.source) else "source"
            }

            argument_cfg.output_dir = temp_output_dir

            initialize_crop_model(
                crop_cfg,
                payload.human_face_detector,
                payload.face_alignment_detector,
                payload.face_alignment_detector_device,
                payload.face_alignment_detector_dtype
            )
            
            download_liveportrait_models()
            if crop_cfg.model == "insightface":
                download_insightface_models()

            retargeting_pipeline = init_retargeting_pipeline(
                inference_cfg,
                crop_cfg,
                argument_cfg,
                payload.use_model_cache
            )

            wfp_concat, wfp = retargeting_pipeline.execute_video_retargeting(
                payload.lip_ratio,
                argument_cfg.source,
                payload.source_face_index,
                payload.retargeting_source_scale,
                payload.driving_smooth_observation_variance_retargeting,
                payload.video_retargeting_silence,
                payload.flag_do_crop_input_retargeting_video
            )
            wfp, wfp_concat = rename_output_videos(wfp, wfp_concat, temp_output_dir, new_names_to_old_names)

            retargeting_video, retargeting_video_with_concat = save_videos_to_ouput(
                wfp,
                wfp_concat,
                payload.output_dir,
                temp_output_dir,
                payload.save_output,
                payload.send_output
            )

            print("Live Portrait API /live-portrait/human/retargeting/video finished")

            return {"retargeting_video": retargeting_video, "retargeting_video_with_concat": retargeting_video_with_concat }
        
        
    @app.post("/live-portrait/animal")
    async def execute_animal(payload: LivePortraitRequest = Body(...)) -> Any:
        print("Live Portrait API /live-portrait/animal received request")
        if IS_MACOS:
            raise OSError("XPose model, required to generate animal videos, is not compatible with MacOS systems.")
        if not has_xpose_lib():
            raise SystemError("The XPose model, required to generate animal videos, is not installed or could not be installed correctly. Try to reinstall it by following instructions in this extension's README.")

        fast_check_inference_args(payload)
        
        argument_cfg = partial_fields(ArgumentConfig, payload.__dict__)
        inference_cfg = partial_fields(InferenceConfig, payload.__dict__)
        crop_cfg = partial_fields(CropConfig, payload.__dict__)

        os.makedirs(temp_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdirname:
            temp_output_dir  = os.path.join(tmpdirname, "output")
            os.makedirs(temp_output_dir, exist_ok=True)

            source_file_extension = get_input_extension(payload.source, payload.source_file_extension)
            driving_file_extension = get_input_extension(payload.driving, payload.driving_file_extension)

            argument_cfg.source = save_input_to_temp_file(payload.source, source_file_extension, tmpdirname)
            argument_cfg.driving = save_input_to_temp_file(payload.driving, driving_file_extension, tmpdirname)

            source_tmp_name = basename(argument_cfg.source)
            driving_tmp_name = basename(argument_cfg.driving)
            new_names_to_old_names = {
                source_tmp_name: basename(payload.source) if is_file(payload.source) else "source",
                driving_tmp_name: basename(payload.driving) if is_file(payload.driving) else "driving"
            }

            argument_cfg.output_dir = temp_output_dir

            download_liveportrait_landmark_model()
            download_liveportrait_animals_models()
            download_insightface_models()

            live_portrait_pipeline_animal = init_live_portrait_animal_pipeline(
                inference_cfg,
                crop_cfg,
                payload.use_model_cache
            )

            wfp, wfp_concat, wfp_gif = live_portrait_pipeline_animal.execute(argument_cfg)
            wfp, wfp_concat, wfp_gif = rename_output_videos_and_gif(wfp, wfp_concat, wfp_gif, temp_output_dir, new_names_to_old_names)

            if payload.output_mode == "images":
                animated_images, animated_images_with_concat = save_video_frames_to_ouput(
                    wfp,
                    wfp_concat,
                    payload.frame_indices,
                    payload.output_dir,
                    temp_output_dir,
                    source_file_extension,
                    payload.save_output,
                    payload.send_output
                )                
                print("Live Portrait API /live-portrait/animal finished")
                return {"animated_images": animated_images, "animated_images_with_concat": animated_images_with_concat }  
            
            animated_image, animated_image_with_concat, animated_gif = save_videos_and_gif_to_ouput(
                wfp,
                wfp_concat,
                wfp_gif,
                payload.output_dir,
                temp_output_dir,
                payload.save_output,
                payload.send_output
            )
            print("Live Portrait API /live-portrait/animal finished")
            return {"animated_video": animated_image, "animated_video_with_concat": animated_image_with_concat }
        

try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(live_portrait_api)
except:
    print("Live Portrait API failed to initialize")
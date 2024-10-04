# coding: utf-8

"""
parameters used for crop faces
"""

import os
from dataclasses import dataclass
from typing import Literal

from .base_config import PrintableConfig, make_abs_path

try:
    from modules.paths_internal import models_path
except:
    try:
        from modules.paths import models_path
    except:
        models_path = os.path.abspath("models")


@dataclass(repr=False)  # use repr from PrintableConfig
class CropConfig(PrintableConfig):
    model: Literal['insightface', 'mediapipe', 'facealignment'] = 'insightface'
    insightface_root: str = os.path.join(models_path, 'insightface')
    landmark_ckpt_path: str = os.path.join(models_path, 'liveportrait', 'landmark.onnx')
    xpose_config_file_path: str = make_abs_path("../utils/dependencies/XPose/config_model/UniPose_SwinT.py")
    xpose_embedding_cache_path: str = make_abs_path('../utils/resources/clip_embedding')
    xpose_ckpt_path: str = os.path.join(models_path, 'liveportrait_animals', 'xpose.pth')
    device_id: int = 0  # gpu device id
    flag_force_cpu: bool = False  # force cpu inference, WIP
    det_thresh: float = 0.15 # detection threshold
    ########## source image or video cropping option ##########
    dsize: int = 512  # crop size
    scale: float = 2.3  # scale factor
    vx_ratio: float = 0  # vx ratio
    vy_ratio: float = -0.125  # vy ratio +up, -down
    max_face_num: int = 0  # max face number, 0 mean no limit
    flag_do_rot: bool = True # whether to conduct the rotation when flag_do_crop is True
    animal_face_type: str = "animal_face_9"  # animal_face_68 -> 68 landmark points, animal_face_9 -> 9 landmarks
    ########## driving video auto cropping option ##########
    scale_crop_driving_video: float = 2.2  # 2.0 # scale factor for cropping driving video
    vx_ratio_crop_driving_video: float = 0.0  # adjust y offset
    vy_ratio_crop_driving_video: float = -0.1  # adjust x offset
    direction: str = "large-small"  # direction of cropping
    ########## face alignment option ##########
    face_alignment_detector: Literal['blazeface', 'blazeface_back_camera', 'retinaface', 'sfd'] = 'blazeface_back_camera'
    face_alignment_detector_device: Literal['cuda', 'cpu', 'mps'] = 'cuda'
    face_alignment_detector_dtype: Literal['fp16', 'bf16', 'fp32'] = 'fp16'
    ########## face index ##########
    source_face_index: int = 0  # source image or video face index
    driving_face_index: int = 0  # driving video face index

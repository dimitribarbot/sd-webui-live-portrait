# coding: utf-8

import os.path as osp
import torch
import numpy as np
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)

from PIL import Image
from typing import List, Tuple, Union
from dataclasses import dataclass, field

from ..config.crop_config import CropConfig
from .crop import (
    average_bbox_lst,
    crop_image,
    crop_image_by_bbox,
    parse_bbox_from_landmark,
)
from .rprint import rlog as log
from .dependencies.face_alignment import FaceAlignment, LandmarksType
from .human_landmark_runner import LandmarkRunner as HumanLandmark


@dataclass
class Trajectory:
    start: int = -1  # start frame
    end: int = -1  # end frame
    lmk_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # lmk list
    bbox_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # bbox list
    M_c2o_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # M_c2o list

    frame_rgb_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame list
    lmk_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # lmk list
    frame_rgb_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame crop list


class Cropper(object):
    def __init__(self, **kwargs) -> None:
        self.crop_cfg: CropConfig = kwargs.get("crop_cfg", None)
        self.image_type = kwargs.get("image_type", 'human_face')
        device_id = kwargs.get("device_id", 0)
        flag_force_cpu = kwargs.get("flag_force_cpu", False)
        face_detector_device = self.crop_cfg.face_alignment_detector_device
        face_detector = self.crop_cfg.face_alignment_detector
        face_detector_dtype = torch.float16
        if self.crop_cfg.face_alignment_detector_dtype == "fp32":
            face_detector_dtype = torch.float32
        elif self.crop_cfg.face_alignment_detector_dtype == "bf16":
            face_detector_dtype = torch.bfloat16
        if flag_force_cpu:
            device = "cpu"
        else:
            try:
                if torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cuda"
            except:
                    device = "cuda"

        self.human_landmark_runner = HumanLandmark(
            ckpt_path=self.crop_cfg.landmark_ckpt_path,
            onnx_provider=device,
            device_id=device_id,
        )
        self.human_landmark_runner.warmup()

        if face_detector == 'blazeface':
            face_detector_kwargs = {'back_model': face_detector_dtype == 'blazeface_back_camera'}
        elif face_detector == 'retinaface':
            face_detector_kwargs = {'fp16': face_detector == torch.float16, 'max_size': 1280}
        else:
            face_detector_kwargs = {}

        self.fa = FaceAlignment(
            LandmarksType.TWO_D,
            flip_input=False,
            device=face_detector_device,
            dtype=face_detector_dtype,
            face_detector=face_detector,
            face_detector_kwargs=face_detector_kwargs
        )

        if self.image_type == "animal_face":
            from .animal_landmark_runner import XPoseRunner as AnimalLandmarkRunner
            self.animal_landmark_runner = AnimalLandmarkRunner(
                    model_config_path=self.crop_cfg.xpose_config_file_path,
                    model_checkpoint_path=self.crop_cfg.xpose_ckpt_path,
                    embeddings_cache_path=self.crop_cfg.xpose_embedding_cache_path,
                    flag_use_half_precision=kwargs.get("flag_use_half_precision", True),
                )
            self.animal_landmark_runner.warmup()

    def update_config(self, user_args):
        for k, v in user_args.items():
            if hasattr(self.crop_cfg, k):
                setattr(self.crop_cfg, k, v)

    def crop_source_image(self, img_rgb_: np.ndarray, crop_cfg: CropConfig):
        # crop a source image and get neccessary information
        img_rgb = img_rgb_.copy()  # copy it

        if self.image_type == "human_face":
            src_face = self.fa.get_landmarks_from_image(img_rgb)

            if src_face is None or len(src_face) == 0:
                log("No face detected in the source image.")
                return None
            elif len(src_face) <= crop_cfg.source_face_index:
                log(f"Only {len(src_face)} faces were detected in the source image. Cannot pick face with index {crop_cfg.source_face_index}.")
                return None
            elif len(src_face) > 1:
                log(f"More than one face detected in the image, only pick one face using face index {crop_cfg.source_face_index}.")

            # NOTE: temporarily only pick the first face, to support multiple face in the future
            src_face = src_face[crop_cfg.source_face_index]

            lmk = np.array(src_face)
        else:
            tmp_dct = {
                'animal_face_9': 'animal_face',
                'animal_face_68': 'face'
            }

            img_rgb_pil = Image.fromarray(img_rgb)
            lmk = self.animal_landmark_runner.run(
                img_rgb_pil,
                'face',
                tmp_dct[crop_cfg.animal_face_type],
                0,
                0
            )

        # crop the face
        ret_dct = crop_image(
            img_rgb,  # ndarray
            lmk,  # 106x2 or Nx2
            dsize=crop_cfg.dsize,
            scale=crop_cfg.scale,
            vx_ratio=crop_cfg.vx_ratio,
            vy_ratio=crop_cfg.vy_ratio,
            flag_do_rot=crop_cfg.flag_do_rot,
        )

        # update a 256x256 version for network input
        ret_dct["img_crop_256x256"] = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
        if self.image_type == "human_face":
            lmk = self.human_landmark_runner.run(img_rgb, lmk)
            ret_dct["lmk_crop"] = lmk
            ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / crop_cfg.dsize
        else:
            # 68x2 or 9x2
            ret_dct["lmk_crop"] = lmk

        return ret_dct

    def calc_lmk_from_cropped_image(self, img_rgb_, **kwargs):
        src_face = self.fa.get_landmarks_from_image(img_rgb_)
        if src_face is None or len(src_face) == 0:
            log("No face detected in the source image.")
            return None
        elif len(src_face) > 1:
            log(f"More than one face detected in the image, only pick one face.")
        src_face = src_face[0]
        lmk = np.array(src_face)
        lmk = self.human_landmark_runner.run(img_rgb_, lmk)

        return lmk

    # TODO: support skipping frame with NO FACE
    def crop_source_video(self, source_rgb_lst, crop_cfg: CropConfig, **kwargs):
        """Tracking based landmarks/alignment and cropping"""
        trajectory = Trajectory()
        for idx, frame_rgb in enumerate(source_rgb_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.fa.get_landmarks_from_image(frame_rgb)
                if src_face is None or len(src_face) == 0:
                    log(f"No face detected in the frame #{idx}")
                    continue
                elif len(src_face) <= crop_cfg.source_face_index:
                    log(f"Only {len(src_face)} faces were detected in the source frame #{idx}. Cannot pick face with index {crop_cfg.source_face_index}.")
                    continue
                elif len(src_face) > 1:
                    log(f"More than one face detected in the source frame_{idx}, only pick one face using face index {crop_cfg.source_face_index}.")
                src_face = src_face[crop_cfg.source_face_index]
                lmk = np.array(src_face)
                lmk = self.human_landmark_runner.run(frame_rgb, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                # TODO: add IOU check for tracking
                lmk = self.human_landmark_runner.run(frame_rgb, trajectory.lmk_lst[-1])
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk)

            # crop the face
            ret_dct = crop_image(
                frame_rgb,  # ndarray
                lmk,  # 106x2 or Nx2
                dsize=crop_cfg.dsize,
                scale=crop_cfg.scale,
                vx_ratio=crop_cfg.vx_ratio,
                vy_ratio=crop_cfg.vy_ratio,
                flag_do_rot=crop_cfg.flag_do_rot,
            )
            lmk = self.human_landmark_runner.run(frame_rgb, lmk)
            ret_dct["lmk_crop"] = lmk

            # update a 256x256 version for network input
            ret_dct["img_crop_256x256"] = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
            ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / crop_cfg.dsize

            trajectory.frame_rgb_crop_lst.append(ret_dct["img_crop_256x256"])
            trajectory.lmk_crop_lst.append(ret_dct["lmk_crop_256x256"])
            trajectory.M_c2o_lst.append(ret_dct['M_c2o'])

        return {
            "frame_crop_lst": trajectory.frame_rgb_crop_lst,
            "lmk_crop_lst": trajectory.lmk_crop_lst,
            "M_c2o_lst": trajectory.M_c2o_lst,
        }

    def crop_driving_video(self, driving_rgb_lst, **kwargs):
        """Tracking based landmarks/alignment and cropping"""
        trajectory = Trajectory()
        for idx, frame_rgb in enumerate(driving_rgb_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.fa.get_landmarks_from_image(frame_rgb)
                if src_face is None or len(src_face) == 0:
                    log(f"No face detected in the frame #{idx}")
                    continue
                elif len(src_face) <= self.crop_cfg.driving_face_index:
                    log(f"Only {len(src_face)} faces were detected in the driving frame #{idx}. Cannot pick face with index {self.crop_cfg.driving_face_index}.")
                    continue
                elif len(src_face) > 1:
                    log(f"More than one face detected in the driving frame_{idx}, only pick one face using face index {self.crop_cfg.driving_face_index}.")
                src_face = src_face[self.crop_cfg.driving_face_index]
                lmk = np.array(src_face)
                lmk = self.human_landmark_runner.run(frame_rgb, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                lmk = self.human_landmark_runner.run(frame_rgb, trajectory.lmk_lst[-1])
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk)
            ret_bbox = parse_bbox_from_landmark(
                lmk,
                scale=self.crop_cfg.scale_crop_driving_video,
                vx_ratio_crop_driving_video=self.crop_cfg.vx_ratio_crop_driving_video,
                vy_ratio=self.crop_cfg.vy_ratio_crop_driving_video,
            )["bbox"]
            bbox = [
                ret_bbox[0, 0],
                ret_bbox[0, 1],
                ret_bbox[2, 0],
                ret_bbox[2, 1],
            ]  # 4,
            trajectory.bbox_lst.append(bbox)  # bbox
            trajectory.frame_rgb_lst.append(frame_rgb)

        global_bbox = average_bbox_lst(trajectory.bbox_lst)

        for idx, (frame_rgb, lmk) in enumerate(zip(trajectory.frame_rgb_lst, trajectory.lmk_lst)):
            ret_dct = crop_image_by_bbox(
                frame_rgb,
                global_bbox,
                lmk=lmk,
                dsize=kwargs.get("dsize", 512),
                flag_rot=False,
                borderValue=(0, 0, 0),
            )
            trajectory.frame_rgb_crop_lst.append(ret_dct["img_crop"])
            trajectory.lmk_crop_lst.append(ret_dct["lmk_crop"])

        return {
            "frame_crop_lst": trajectory.frame_rgb_crop_lst,
            "lmk_crop_lst": trajectory.lmk_crop_lst,
        }


    def calc_lmks_from_cropped_video(self, driving_rgb_crop_lst, **kwargs):
        """Tracking based landmarks/alignment"""
        trajectory = Trajectory()

        for idx, frame_rgb_crop in enumerate(driving_rgb_crop_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.fa.get_landmarks_from_image(frame_rgb_crop)
                if src_face is None or len(src_face) == 0:
                    log(f"No face detected in the frame #{idx}")
                    raise Exception(f"No face detected in the frame #{idx}")
                elif len(src_face) > 1:
                    log(f"More than one face detected in the driving frame_{idx}, only pick one face.")
                src_face = src_face[0]
                lmk = np.array(src_face)
                lmk = self.human_landmark_runner.run(frame_rgb_crop, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                lmk = self.human_landmark_runner.run(frame_rgb_crop, trajectory.lmk_lst[-1])
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk)
        return trajectory.lmk_lst

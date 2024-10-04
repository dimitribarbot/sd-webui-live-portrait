import os
from typing import Any, Dict

from torch.hub import download_url_to_file

from ..core import FaceDetector

try:
    from modules.paths_internal import models_path
except:
    try:
        from modules.paths import models_path
    except:
        models_path = os.path.abspath("models")

model_dir = os.path.join(models_path, "FaceAlignment")

models_urls = {
    'retinaface': 'https://github.com/dimitribarbot/sd-webui-live-portrait/releases/download/v0.2.4/Resnet50_Final.safetensors',
}

from liveportrait.utils.dependencies.face_alignment.detection.retinaface.alignment import load_net, batch_detect


class RetinaFaceDetector(FaceDetector):
    '''RetinaFace Detector.
    '''

    def __init__(self, device, path_to_detector=None, verbose=False, threshold=0.5, fp16=True):
        super(RetinaFaceDetector, self).__init__(device, verbose)

        # Initialise the face detector
        if path_to_detector is None:
            path_to_detector = self.load_from_url(models_urls['retinaface'], model_dir)

        self.threshold = threshold
        self.fp16 = fp16

        self.face_detector = load_net(path_to_detector, self.device)
        if self.fp16:
            self.face_detector.half()

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        detected_faces = batch_detect(
            self.face_detector,
            [image],
            threshold=self.threshold,
            return_dict=True,
            fp16=self.fp16,
            device=self.device
        )[0]
        bboxlist = [detected_face["box"] for detected_face in detected_faces]

        return bboxlist

    def detect_from_batch(self, tensor):
        all_detected_faces = batch_detect(
            self.face_detector,
            tensor,
            is_tensor=True,
            threshold=self.threshold,
            return_dict=True,
            fp16=self.fp16,
            device=self.device
        )

        all_bboxlists = []
        for detected_faces in all_detected_faces:
            bboxlist = [detected_face["box"] for detected_face in detected_faces]
            all_bboxlists.append(bboxlist)

        return all_bboxlists
    
    @classmethod
    def load_from_url(cls, url: str, model_dir: str) -> Dict[str, Any]:
        import sys
        from urllib.parse import urlparse
        os.makedirs(model_dir, exist_ok=True)
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            hash_prefix = None
            download_url_to_file(url, cached_file, hash_prefix, progress=True)
        return cached_file

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
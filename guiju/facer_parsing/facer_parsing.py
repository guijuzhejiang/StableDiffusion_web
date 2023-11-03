# coding=utf-8
# @Time : 2023/11/3 下午12:55
# @File : facer_parsing.py
import numpy as np
import torch
import facer
import os

from PIL import Image


class FaceParsing:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_dir = os.path.join(os.path.dirname(__file__), 'models')

    def __init__(self):
        self.face_detector = facer.face_detector('retinaface/mobilenet', device=self.device,
                                            model_path=os.path.join(self.model_dir, 'mobilenet0.25_Final.pth'))

        self.face_parser = facer.face_parser('farl/lapa/448', device=self.device,
                                        model_path=os.path.join(self.model_dir, 'face_parsing.farl.lapa.main_ema_136500_jit191.pt'))  # optional "farl/celebm/448"

    def __call__(self, _input_pil_image, keep='face', *args, **kwargs):
        image = facer.hwc2bchw(torch.from_numpy(np.array(_input_pil_image.convert('RGB')))).to(device=self.device)  # image: 1 x 3 x h x w

        with torch.inference_mode():
            faces = self.face_detector(image)
            if faces:
                faces = self.face_parser(image, faces)
            else:
                return None

        seg_logits = faces['seg']['logits'][0]
        seg_probs = seg_logits.softmax(dim=0)
        seg_labels = seg_probs.argmax(dim=0).cpu().numpy()

        if keep == 'face':
            seg_labels[(seg_labels == 0) | (seg_labels == 10)] = 0

        else:
            seg_labels[seg_labels != 10] = 0

        width, height = seg_labels.shape[1], seg_labels.shape[0]
        image = Image.new("L", (width, height), 0)
        for y in range(height):
            for x in range(width):
                if seg_labels[y, x] != 0:
                    image.putpixel((x, y), 255)  # 0表示黑色
        return image

    def detect_face(self, _input_pil_image):
        image = facer.hwc2bchw(torch.from_numpy(np.array(_input_pil_image.convert('RGB')))).to(device=self.device)  # image: 1 x 3 x h x w

        with torch.inference_mode():
            res = self.face_detector(image)
        if res:
            return [[int(i) for i in x] for x in res['rects'].cpu().numpy().tolist()]
        else:
            return []

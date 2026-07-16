from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from dataset import get_data_transforms
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2


PROJECT_ROOT = Path(
    "/home/jooyeon/ros2_ws/src/doosan-robot2"
)

VALID_PATH = (
    PROJECT_ROOT
    / "rd_dataset_crop"
    / "valid"
    / "images"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "wres50_cube_crop_best.pth"
)

IMAGE_SIZE = 256

# 정상 validation 점수의 상위 99% 지점
THRESHOLD_PERCENTILE = 99

# anomaly map에서 상위 1% 픽셀의 평균
TOP_PIXEL_RATIO = 0.01


def crop_cube_by_bbox(
    image,
    bbox,
    padding_ratio=0.15,
):
    x1, y1, x2, y2 = bbox

    image_height, image_width = (
        image.shape[:2]
    )

    box_width = x2 - x1
    box_height = y2 - y1

    padding_x = int(
        box_width * padding_ratio
    )

    padding_y = int(
        box_height * padding_ratio
    )

    crop_x1 = max(
        0,
        int(x1) - padding_x,
    )

    crop_y1 = max(
        0,
        int(y1) - padding_y,
    )

    crop_x2 = min(
        image_width,
        int(x2) + padding_x,
    )

    crop_y2 = min(
        image_height,
        int(y2) + padding_y,
    )

    crop = image[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ].copy()

    if crop.size == 0:
        raise RuntimeError(
            "큐브 Crop 결과가 비어 있습니다."
        )

    crop = cv2.resize(
        crop,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    return crop


def calculate_robust_anomaly_score(
    anomaly_map,
    top_ratio=0.01,
):
    values = anomaly_map.reshape(-1)

    top_count = max(
        1,
        int(values.size * top_ratio),
    )

    top_values = np.partition(
        values,
        -top_count,
    )[-top_count:]

    return float(
        np.mean(top_values)
    )


class RDInspector:
    def __init__(self):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "RD device:",
            self.device,
        )

        self.transform = get_data_transforms(
            IMAGE_SIZE,
            IMAGE_SIZE,
        )

        (
            self.encoder,
            self.bn,
            self.decoder,
        ) = self._load_model()

        self.threshold = (
            self._calculate_threshold()
        )

        print(
            f"RD threshold: "
            f"{self.threshold:.6f}"
        )

    def _load_model(self):
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"RD 체크포인트가 없습니다:\n"
                f"{CHECKPOINT_PATH}"
            )

        encoder, bn = wide_resnet50_2(
            pretrained=True
        )

        decoder = de_wide_resnet50_2(
            pretrained=False
        )

        encoder = encoder.to(
            self.device
        )

        bn = bn.to(
            self.device
        )

        decoder = decoder.to(
            self.device
        )

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
        )

        bn_state_dict = (
            checkpoint["bn"].copy()
        )

        for key in list(
            bn_state_dict.keys()
        ):
            if "memory" in key:
                bn_state_dict.pop(key)

        bn.load_state_dict(
            bn_state_dict,
            strict=True,
        )

        decoder.load_state_dict(
            checkpoint["decoder"],
            strict=True,
        )

        encoder.eval()
        bn.eval()
        decoder.eval()

        print(
            "RD model loaded:",
            CHECKPOINT_PATH,
        )

        return encoder, bn, decoder

    def _calculate_anomaly_map(
        self,
        encoder_features,
        decoder_features,
    ):
        anomaly_map = np.zeros(
            (IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.float32,
        )

        for encoder_feature, decoder_feature in zip(
            encoder_features,
            decoder_features,
        ):
            layer_map = (
                1
                - F.cosine_similarity(
                    encoder_feature,
                    decoder_feature,
                    dim=1,
                )
            )

            layer_map = (
                layer_map.unsqueeze(1)
            )

            layer_map = F.interpolate(
                layer_map,
                size=(
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                mode="bilinear",
                align_corners=True,
            )

            layer_map = (
                layer_map[0, 0]
                .detach()
                .cpu()
                .numpy()
            )

            anomaly_map += layer_map

        return anomaly_map

    def _calculate_tensor_score(
        self,
        image_tensor,
    ):
        if image_tensor.dim() == 3:
            image_tensor = (
                image_tensor.unsqueeze(0)
            )

        image_tensor = image_tensor.to(
            self.device
        )

        with torch.no_grad():
            encoder_features = (
                self.encoder(
                    image_tensor
                )
            )

            decoder_features = (
                self.decoder(
                    self.bn(
                        encoder_features
                    )
                )
            )

            anomaly_map = (
                self._calculate_anomaly_map(
                    encoder_features,
                    decoder_features,
                )
            )

        anomaly_map = gaussian_filter(
            anomaly_map,
            sigma=4,
        )

        score = (
            calculate_robust_anomaly_score(
                anomaly_map,
                top_ratio=TOP_PIXEL_RATIO,
            )
        )

        return score, anomaly_map

    def _calculate_threshold(self):
        if not VALID_PATH.exists():
            raise FileNotFoundError(
                f"Validation 경로가 없습니다:\n"
                f"{VALID_PATH}"
            )

        valid_dataset = ImageFolder(
            root=str(VALID_PATH),
            transform=self.transform,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        valid_scores = []

        for images, _ in valid_loader:
            score, _ = (
                self._calculate_tensor_score(
                    images
                )
            )

            valid_scores.append(
                score
            )

        valid_scores = np.array(
            valid_scores,
            dtype=np.float32,
        )

        threshold = float(
            np.percentile(
                valid_scores,
                THRESHOLD_PERCENTILE,
            )
        )

        print("\n===== Validation score =====")
        print("count:", len(valid_scores))
        print(
            f"min: {valid_scores.min():.6f}"
        )
        print(
            f"mean: {valid_scores.mean():.6f}"
        )
        print(
            f"max: {valid_scores.max():.6f}"
        )
        print(
            f"threshold: {threshold:.6f}"
        )

        return threshold

    def predict_cv_image(
        self,
        bgr_image,
    ):
        if bgr_image is None:
            raise ValueError(
                "검사 이미지가 없습니다."
            )

        if bgr_image.size == 0:
            raise ValueError(
                "검사 이미지 크기가 0입니다."
            )

        resized_image = cv2.resize(
            bgr_image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

        rgb_image = cv2.cvtColor(
            resized_image,
            cv2.COLOR_BGR2RGB,
        )

        pil_image = Image.fromarray(
            rgb_image
        )

        image_tensor = self.transform(
            pil_image
        )

        score, anomaly_map = (
            self._calculate_tensor_score(
                image_tensor
            )
        )

        is_defective = (
            score >= self.threshold
        )

        prediction = (
            "불량 의심"
            if is_defective
            else "정상 판정"
        )

        return {
            "prediction": prediction,
            "is_defective": is_defective,
            "score": score,
            "threshold": self.threshold,
            "anomaly_map": anomaly_map,
        }
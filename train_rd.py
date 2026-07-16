import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from dataset import get_data_transforms
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2


PROJECT_ROOT = Path(
    "/home/jooyeon/ros2_ws/src/doosan-robot2"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "rd_dataset_crop"
    / "train"
    / "images"
)

VALID_PATH = (
    PROJECT_ROOT
    / "rd_dataset_crop"
    / "valid"
    / "images"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "wres50_cube_crop_best.pth"
)

FINAL_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "wres50_cube_crop_final.pth"
)

EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.005
IMAGE_SIZE = 256
SEED = 111
NUM_WORKERS = 0


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_dataset(
    path,
    name,
):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} 경로가 없습니다:\n{path}"
        )

    dataset = ImageFolder(
        root=str(path)
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"{name} 이미지가 없습니다."
        )

    print(
        f"{name} 이미지 수:",
        len(dataset),
    )

    print(
        f"{name} 클래스:",
        dataset.class_to_idx,
    )


def loss_function(
    encoder_features,
    decoder_features,
):
    cosine_similarity = (
        torch.nn.CosineSimilarity(
            dim=1
        )
    )

    loss = torch.zeros(
        (),
        device=encoder_features[0].device,
    )

    for encoder_feature, decoder_feature in zip(
        encoder_features,
        decoder_features,
    ):
        encoder_flatten = (
            encoder_feature.reshape(
                encoder_feature.shape[0],
                -1,
            )
        )

        decoder_flatten = (
            decoder_feature.reshape(
                decoder_feature.shape[0],
                -1,
            )
        )

        loss += torch.mean(
            1
            - cosine_similarity(
                encoder_flatten,
                decoder_flatten,
            )
        )

    return loss


def train_one_epoch(
    encoder,
    bn,
    decoder,
    dataloader,
    optimizer,
    device,
):
    encoder.eval()
    bn.train()
    decoder.train()

    losses = []

    for images, _ in dataloader:
        images = images.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.no_grad():
            encoder_features = encoder(
                images
            )

        decoder_features = decoder(
            bn(encoder_features)
        )

        loss = loss_function(
            encoder_features,
            decoder_features,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "학습 loss가 NaN 또는 Inf입니다."
            )

        loss.backward()
        optimizer.step()

        losses.append(
            loss.item()
        )

    return float(
        np.mean(losses)
    )


def validate_one_epoch(
    encoder,
    bn,
    decoder,
    dataloader,
    device,
):
    encoder.eval()
    bn.eval()
    decoder.eval()

    losses = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(
                device,
                non_blocking=True,
            )

            encoder_features = encoder(
                images
            )

            decoder_features = decoder(
                bn(encoder_features)
            )

            loss = loss_function(
                encoder_features,
                decoder_features,
            )

            losses.append(
                loss.item()
            )

    return float(
        np.mean(losses)
    )


def save_checkpoint(
    save_path,
    epoch,
    bn,
    decoder,
    optimizer,
    train_loss,
    valid_loss,
):
    torch.save(
        {
            "epoch": epoch,
            "image_size": IMAGE_SIZE,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "bn": bn.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        save_path,
    )

    print(
        "모델 저장:",
        save_path,
    )


def train():
    validate_dataset(
        TRAIN_PATH,
        "Train",
    )

    validate_dataset(
        VALID_PATH,
        "Valid",
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n===== RD Crop 모델 학습 =====")
    print("Device:", device)
    print("Train:", TRAIN_PATH)
    print("Valid:", VALID_PATH)
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    transform = get_data_transforms(
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    train_dataset = ImageFolder(
        root=str(TRAIN_PATH),
        transform=transform,
    )

    valid_dataset = ImageFolder(
        root=str(VALID_PATH),
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    encoder, bn = wide_resnet50_2(
        pretrained=True
    )

    decoder = de_wide_resnet50_2(
        pretrained=False
    )

    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)

    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.Adam(
        list(bn.parameters())
        + list(decoder.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    best_valid_loss = float("inf")
    total_start_time = time.time()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        epoch_start_time = time.time()

        train_loss = train_one_epoch(
            encoder=encoder,
            bn=bn,
            decoder=decoder,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        valid_loss = validate_one_epoch(
            encoder=encoder,
            bn=bn,
            decoder=decoder,
            dataloader=valid_loader,
            device=device,
        )

        elapsed = (
            time.time()
            - epoch_start_time
        )

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"train={train_loss:.6f} "
            f"valid={valid_loss:.6f} "
            f"time={elapsed:.2f}s"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            save_checkpoint(
                save_path=BEST_CHECKPOINT_PATH,
                epoch=epoch,
                bn=bn,
                decoder=decoder,
                optimizer=optimizer,
                train_loss=train_loss,
                valid_loss=valid_loss,
            )

        if epoch % 10 == 0:
            epoch_path = (
                CHECKPOINT_DIR
                / f"wres50_cube_crop_epoch_{epoch}.pth"
            )

            save_checkpoint(
                save_path=epoch_path,
                epoch=epoch,
                bn=bn,
                decoder=decoder,
                optimizer=optimizer,
                train_loss=train_loss,
                valid_loss=valid_loss,
            )

    save_checkpoint(
        save_path=FINAL_CHECKPOINT_PATH,
        epoch=EPOCHS,
        bn=bn,
        decoder=decoder,
        optimizer=optimizer,
        train_loss=train_loss,
        valid_loss=valid_loss,
    )

    print("\n===== 학습 완료 =====")
    print(
        "Best validation loss:",
        best_valid_loss,
    )
    print(
        "Best model:",
        BEST_CHECKPOINT_PATH,
    )
    print(
        "전체 시간:",
        time.time() - total_start_time,
    )


if __name__ == "__main__":
    warnings.simplefilter(
        action="ignore",
        category=FutureWarning,
    )

    setup_seed(SEED)
    train()
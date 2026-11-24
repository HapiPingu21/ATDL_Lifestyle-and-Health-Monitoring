"""Run a pinned Food-101 classifier entirely on the local CPU."""

from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
MODEL_ID = "mobilenet-v2-food101@7d05d38"
MODEL_PATH = ROOT / ".data" / "food-model" / "pytorch_model.bin"
MODEL_URL = (
    "https://huggingface.co/AlexKoff88/mobilenet_v2_food101/resolve/"
    "7d05d3878ee7f2aad7ad159018a8e5fdb4953d41/pytorch_model.bin"
)
MODEL_SHA256 = "42cd5d9988f0830588b3be5107197c9b622c9b73426377975c07e8b24e346680"
MODEL_BYTES = 19_121_039
PHOTO_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
MAX_PHOTO_BYTES = 8 * 1024 * 1024
MAX_PHOTO_PIXELS = 24_000_000


class PhotoUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_classifier():
    if not MODEL_PATH.is_file():
        raise PhotoUnavailable("Run python food_photo.py to download the local model.")
    import torch
    from torchvision import transforms
    from torchvision.models import mobilenet_v2

    weights = MODEL_PATH.read_bytes()
    if sha256(weights).hexdigest() != MODEL_SHA256:
        raise PhotoUnavailable("The model checksum is invalid. Run python food_photo.py.")
    labels = (ROOT / "food101_labels.txt").read_text(encoding="utf-8").splitlines()
    if len(labels) != 101 or labels != sorted(set(labels)):
        raise PhotoUnavailable("The Food-101 class list is invalid.")

    torch.set_num_threads(2)
    checkpoint = torch.load(BytesIO(weights), map_location="cpu", weights_only=True)
    model = mobilenet_v2(num_classes=len(labels))
    model.load_state_dict({
        key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()
    })
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model.eval(), preprocess, labels


def suggest_food(photo, content_type):
    try:
        import torch
        from PIL import Image, ImageOps
        model, preprocess, labels = load_classifier()
    except ImportError as error:
        raise PhotoUnavailable("Install the packages in requirements-photo.txt.") from error

    try:
        with Image.open(BytesIO(photo), formats=list(PHOTO_TYPES.values())) as image:
            if image.format != PHOTO_TYPES[content_type]:
                raise ValueError("The photo contents do not match its file type.")
            if image.width * image.height > MAX_PHOTO_PIXELS:
                raise ValueError("Choose a photo with at most 24 megapixels.")
            rgb = ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Choose a valid JPEG, PNG, or WebP photo.") from error

    scale = 256 / min(rgb.size)
    width, height = (int(dimension * scale) for dimension in rgb.size)
    if width * height > MAX_PHOTO_PIXELS:
        raise ValueError("Choose a photo with a less extreme aspect ratio.")
    with torch.inference_mode():
        scores, indices = model(preprocess(rgb)[None]).softmax(dim=1)[0].topk(3)
    return {
        "model": MODEL_ID,
        "suggestions": [
            {"food": labels[index].replace("_", " "), "score": round(score, 4)}
            for index, score in zip(indices.tolist(), scores.tolist())
        ],
    }


def download_model():
    """Explicit setup step. Inference never makes network requests."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(MODEL_URL, timeout=30) as response:
        weights = response.read(MODEL_BYTES + 1)
    if len(weights) != MODEL_BYTES or sha256(weights).hexdigest() != MODEL_SHA256:
        raise ValueError("The downloaded model did not match the pinned file.")
    with NamedTemporaryFile(dir=MODEL_PATH.parent, delete=False) as target:
        temporary = Path(target.name)
    try:
        temporary.write_bytes(weights)
        temporary.replace(MODEL_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Local food-photo model ready at {MODEL_PATH}")


if __name__ == "__main__":
    download_model()

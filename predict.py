import sys
from pathlib import Path
from PIL import Image

from model_engine import MaskClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_mask_model.h5"
DEFAULT_IMAGE_PATH = BASE_DIR / "test_image.jpg"


def main():
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    classifier = MaskClassifier(MODEL_PATH)
    with Image.open(image_path) as img:
        result = classifier.predict_image(img)

    print(f"Image: {image_path}")
    print(f"Prediction: {result['label']} ({result['prediction']})")
    print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()

import cv2
import numpy as np
from .base_extractor import BaseFeatureExtractor

class SIFTExtractor(BaseFeatureExtractor):
    IMAGE_FORMATS = [
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
        '.webp', '.jp2', '.dib', '.pbm', '.pgm', '.ppm'
    ]

    def __init__(
        self,
        n_features: int = 128,
    ):
        super().__init__()
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
        )

    def extract(self, file_path: str) -> np.ndarray:
        self.archivo_valido(file_path)

        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"No se pudo leer la imagen: {file_path}")

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        _, descriptors = self.sift.detectAndCompute(gray, None)

        if descriptors is None or len(descriptors) == 0:
            raise RuntimeError(
                f"No existen descriptores SIFT en la imagen: {file_path}"
            )

        return descriptors.mean(axis=0).astype(np.float32)

    def get_feature_dimension(self) -> int:
        return 128

    def supported_formats(self) -> list:
        return self.IMAGE_FORMATS

    def __str__(self) -> str:
        return f"SIFTExtractor(n_features={self.sift.getNFeatures()})"

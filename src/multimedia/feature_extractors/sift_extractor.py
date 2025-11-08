import cv2
import numpy as np
from .base_extractor import BaseFeatureExtractor

class SIFTExtractor(BaseFeatureExtractor):
    IMAGE_FORMATS = [
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
        '.webp', '.jp2', '.dib', '.pbm', '.pgm', '.ppm'
    ]

    def __init__(self):
        super().__init__()
        self.sift = cv2.SIFT_create()

    def extract(self, file_path: str) -> np.ndarray:
        self.archivo_valido(file_path)

        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"No se pudo leer la imagen: {file_path}")


        # En caso que la imagen sea muy grande, redimensionar para poder procesar más rapido
        max_dim = 800
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # SIFT Solo funciona con imágenes en escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        _, descriptors = self.sift.detectAndCompute(gray, None)

        if descriptors is None or len(descriptors) == 0:
            raise RuntimeError(f"No existen descriptores SIFT en la imagen: {file_path}")

        return descriptors.astype(np.float32)

    def get_feature_dimension(self) -> int:
        return 128

    def supported_formats(self) -> list:
        return self.IMAGE_FORMATS

    def __str__(self) -> str:
        return "SIFTExtractor"

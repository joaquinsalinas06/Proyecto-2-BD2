import librosa
import numpy as np
from typing import Optional
from .base_extractor import BaseFeatureExtractor

class MFCCExtractor(BaseFeatureExtractor):
    AUDIO_FORMATS = [
        '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.wma',
        '.aac', '.opus', '.aiff', '.au'
    ]

    def __init__(
        self,
    ):
        super().__init__()

    def extract(self, file_path: str) -> np.ndarray:
        self.archivo_valido(file_path)

        try:
            y, sr = librosa.load(
                file_path,
                mono=True
            )

            if len(y) == 0:
                raise ValueError(f"El archivo de audio está vacío: {file_path}")
            
            mfccs = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=25
            )
            return mfccs.mean(axis=1).astype(np.float32)

        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            raise RuntimeError(f"Hubo un error al extraer MFCC de {file_path}: {e}") from e

    def get_feature_dimension(self) -> int:
        return 25

    def supported_formats(self) -> list:
        return self.AUDIO_FORMATS

    def __str__(self) -> str:
        return f"MFCCExtractor(n_mfcc=25)"

from abc import ABC, abstractmethod
import numpy as np
import os

class BaseFeatureExtractor(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def extract(self, file_path: str) -> np.ndarray:
        pass

    @abstractmethod
    def get_feature_dimension(self) -> int:
        pass

    @abstractmethod
    def supported_formats(self) -> list:
        pass

    def archivo_valido(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext not in self.supported_formats():
            raise ValueError(
                f"Formato de archivo no soportado: {ext}. "
            )

        return True
    
    def __str__(self) -> str:
        return f"{self.name}(dimensión={self.get_feature_dimension()}, formatos={self.supported_formats()})"

    def __repr__(self) -> str:
        return self.__str__()

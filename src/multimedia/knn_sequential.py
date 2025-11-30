"""
KNN Sequential - Búsqueda secuencial lineal para imágenes.
Implementa búsqueda KNN comparando con todos los vectores de características.
"""
import struct
import numpy as np
from typing import List, Tuple


class KNNSequential:
    """
    Índice KNN Secuencial que realiza búsqueda lineal sobre todos los vectores.
    """
    
    def __init__(self, histograms_file: str, mapping_file: str):
        """
        Inicializa el índice KNN secuencial.
        
        Args:
            histograms_file: Ruta al archivo de histogramas TF-IDF
            mapping_file: Ruta al archivo de mapeo doc_id -> product_id
        """
        self.histograms_file = histograms_file
        self.mapping_file = mapping_file
        self.histograms = None
        self.mapping = None
        self.num_docs = 0
        self.vocab_size = 0
    
    def load_index(self):
        """Carga el índice desde disco."""
        # Cargar histogramas
        with open(self.histograms_file, 'rb') as f:
            self.num_docs, self.vocab_size = struct.unpack('ii', f.read(8))
            
            histograms = []
            for _ in range(self.num_docs):
                hist_bytes = f.read(self.vocab_size * 4)  # 4 bytes por float32
                hist = np.frombuffer(hist_bytes, dtype=np.float32)
                histograms.append(hist)
            
            self.histograms = np.array(histograms)
        
        # Cargar mapping
        mapping = []
        with open(self.mapping_file, 'rb') as f:
            num_docs, id_size, path_size = struct.unpack('iii', f.read(12))
            
            for _ in range(num_docs):
                id_bytes = f.read(id_size)
                path_bytes = f.read(path_size)
                
                product_id = id_bytes.rstrip(b'\x00').decode('utf-8')
                mapping.append(product_id)
        
        self.mapping = mapping
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula la similitud coseno entre dos vectores."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def search(self, query_vector: np.ndarray, k: int = 8) -> List[Tuple[int, float]]:
        """
        Realiza búsqueda KNN secuencial.
        
        Args:
            query_vector: Vector de características de la query
            k: Número de vecinos más cercanos a retornar
            
        Returns:
            Lista de tuplas (doc_id, similitud) ordenadas por similitud descendente
        """
        if self.histograms is None:
            raise RuntimeError("Debe llamar a load_index() antes de realizar búsquedas")
        
        similarities = []
        for idx, hist in enumerate(self.histograms):
            sim = self._cosine_similarity(query_vector, hist)
            similarities.append((idx, sim))
        
        # Ordenar por similitud descendente y tomar top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def get_vector(self, doc_id: int) -> np.ndarray:
        """Retorna el vector de características de un documento."""
        if self.histograms is None:
            raise RuntimeError("Debe llamar a load_index() antes de acceder a vectores")
        return self.histograms[doc_id]
    
    def get_product_id(self, doc_id: int) -> str:
        """Retorna el product_id de un documento."""
        if self.mapping is None:
            raise RuntimeError("Debe llamar a load_index() antes de acceder al mapping")
        return self.mapping[doc_id]
    
    def get_num_docs(self) -> int:
        """Retorna el número de documentos indexados."""
        return self.num_docs
    
    def get_vocab_size(self) -> int:
        """Retorna el tamaño del vocabulario."""
        return self.vocab_size

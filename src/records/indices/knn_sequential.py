from typing import List, Dict, Any, Tuple
import numpy as np
import heapq
import os
import struct
from .base_index import MultimediaIndexBase
from src.multimedia.codebook import Codebook
from src.multimedia.feature_extractors import SIFTExtractor, MFCCExtractor


class KNNSequentialIndex(MultimediaIndexBase):
    def __init__(self, column_name: str, table_schema, filename: str = None,
                 is_primary: bool = False, primary_key_column: str = None,
                 vocabulary_size: int = 100, codebook_file: str = None,
                 extractor=None):
        super().__init__(column_name, filename)
        self.table_schema = table_schema
        self.vocabulary_size = vocabulary_size
        self.codebook_file = codebook_file
        self.extractor = extractor
 
        self.is_primary = is_primary
        self.primary_key_column = primary_key_column
        if not self.extractor:
            col = None
            for c in table_schema:
                if c.name == column_name:
                    col = c
                    break

            if col:
                if col.data_type.value == "IMAGE":
                    self.extractor = SIFTExtractor()
                elif col.data_type.value == "AUDIO":
                    self.extractor = MFCCExtractor()

        # Archivos de mapping e histogramas pre-calculados
        if codebook_file and os.path.exists(codebook_file):
            base = codebook_file.replace('_codebook.dat', '')
            self.mapping_file = f"{base}_index_mapping.dat"
            self.histograms_file = f"{base}_index_histograms.dat"
        else:
            self.mapping_file = None
            self.histograms_file = None

        self.total_docs = 0
        self.id_size = 16
        self.path_size = 240
        self.fixed_path_size = 256
        self._load_mapping()

        data_type = "IMAGE"
        for c in table_schema:
            if c.name == column_name:
                data_type = c.data_type.value
                break

        self.codebook = Codebook(k=vocabulary_size, descriptor_type=data_type,
                                codebook_file=self.codebook_file)
        if codebook_file and os.path.exists(self.codebook_file):
            self.codebook.load_from_disk()

    # Se cargan datos como la cantidad de documentos, el tamaño del ID y del path
    def _load_mapping(self):
        if not self.mapping_file or not os.path.exists(self.mapping_file):
            self.total_docs = 0
            return

        with open(self.mapping_file, 'rb') as f:
            header = f.read(12)
            if len(header) < 12:
                self.total_docs = 0
                return

            self.total_docs, self.id_size, self.path_size = struct.unpack('iii', header)
            self.fixed_path_size = self.id_size + self.path_size

    # Sabiendo cuanto es el offset obtenemos el path del documento
    def _get_path_by_doc_id(self, doc_id: int) -> str:
        offset = 12 + (doc_id * self.fixed_path_size)

        with open(self.mapping_file, 'rb') as f:
            f.seek(offset + self.id_size)
            path_bytes = f.read(self.path_size)

        path = path_bytes.decode('utf-8', errors='ignore').rstrip('\x00')
        return path

    # En los casos donde tengamos un id mapeado lo extraemos
    def _get_id_by_doc_id(self, doc_id: int) -> int:
        offset = 12 + (doc_id * self.fixed_path_size)

        with open(self.mapping_file, 'rb') as f:
            f.seek(offset)
            id_bytes = f.read(self.id_size)

        id_str = id_bytes.decode('utf-8', errors='ignore').rstrip('\x00')

        try:
            return int(id_str)
        except ValueError:
            return id_str

    def knnSearch(self, query_vector: np.ndarray, k: int) -> List[Tuple[Dict[str, Any], float]]:
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)

        max_heap = []

        # ESTRATEGIA: Escaneo secuencial de todos los documentos
        if self.histograms_file and os.path.exists(self.histograms_file):
            with open(self.histograms_file, 'rb') as f:
                num_docs, vocab_size = struct.unpack('ii', f.read(8))
                histogram_size_bytes = vocab_size * 4

                for doc_id in range(min(num_docs, self.total_docs)):
                    histogram_bytes = f.read(histogram_size_bytes)
                    if len(histogram_bytes) < histogram_size_bytes:
                        break

                    doc_vector = np.frombuffer(histogram_bytes, dtype=np.float32)
                    distance = self._cosine_distance(query_vector, doc_vector)

                    if len(max_heap) < k:
                        heapq.heappush(max_heap, (-distance, doc_id, doc_vector))
                    elif distance < -max_heap[0][0]:
                        heapq.heapreplace(max_heap, (-distance, doc_id, doc_vector))

        else:
            for doc_id in range(self.total_docs):
                file_path = self._get_path_by_doc_id(doc_id)
                if not file_path:
                    continue

                try:
                    descriptors = self.extractor.extract(file_path)
                    doc_vector = self.codebook.descriptors_to_histogram(descriptors, apply_tfidf=True)
                    distance = self._cosine_distance(query_vector, doc_vector)

                    if len(max_heap) < k:
                        heapq.heappush(max_heap, (-distance, doc_id, doc_vector))
                    elif distance < -max_heap[0][0]:
                        heapq.heapreplace(max_heap, (-distance, doc_id, doc_vector))
                except Exception:
                    continue

        results = []
        pk_column_name = self.primary_key_column if self.primary_key_column else 'id'

        for neg_dist, doc_id, doc_vector in max_heap:
            product_id = self._get_id_by_doc_id(doc_id)
            file_path = self._get_path_by_doc_id(doc_id)

            if product_id is None or not file_path:
                continue

            record = {
                pk_column_name: product_id,
                self.column_name: {
                    'path': file_path
                }
            }
            results.append((record, -neg_dist))

        results.sort(key=lambda x: x[1])
        return results

    def knnSearchByFile(self, file_path: str, k: int) -> List[Tuple[Dict[str, Any], float]]:
        if not os.path.exists(self.codebook_file):
            raise RuntimeError(f"Codebook no encontrado: {self.codebook_file}")

        if self.codebook.codewords is None:
            self.codebook.load_from_disk()

        # Extraemos features del archivo de consulta
        descriptors = self.extractor.extract(file_path)
        query_vector = self.codebook.descriptors_to_histogram(descriptors, apply_tfidf=True)

        return self.knnSearch(query_vector, k)

    def _cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 1.0

        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(1.0 - similarity)

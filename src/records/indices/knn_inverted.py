from typing import List, Dict, Any, Tuple
import numpy as np
import heapq
import os
import struct
from .base_index import MultimediaIndexBase
from ..record import DynamicRecord
from src.multimedia.codebook import Codebook


class KNNInvertedIndex(MultimediaIndexBase):
    def __init__(self, column_name: str, table_schema, filename: str = None,
                 is_primary: bool = False, primary_key_column: str = None,
                 vocabulary_size: int = 100, codebook_file: str = None,
                 extractor=None, idf_threshold: float = 0.0, index_prefix: str = None):
        super().__init__(column_name, filename)
        self.table_schema = table_schema
        self.vocabulary_size = vocabulary_size
        self.idf_threshold = idf_threshold

        self.is_primary = is_primary
        self.primary_key_column = primary_key_column

        temp_format = DynamicRecord._build_format(table_schema)
        self.record_size = struct.calcsize(temp_format)

        if filename:
            if not filename.endswith('.dat'):
                filename = filename + '.dat'
            self.data_file = filename
        else:
            self.data_file = f"{column_name}_knn_inverted_tfidf.dat"

        if index_prefix:
            indices_dir = os.path.dirname(codebook_file) if codebook_file else os.path.dirname(self.data_file)
            base = os.path.join(indices_dir, index_prefix)
            self.inverted_index_file = f"{base}_inverted.dat"
            self.mapping_file = f"{base}_mapping.dat"
            self.histograms_file = f"{base}_histograms.dat"
        elif codebook_file and os.path.exists(codebook_file):
            import re
            base = re.sub(r'_codebook\.dat$', '', codebook_file)
            self.inverted_index_file = f"{base}_inverted.dat"
            self.mapping_file = f"{base}_mapping.dat"
            self.histograms_file = f"{base}_histograms.dat"
        else:
            self.inverted_index_file = self.data_file.replace('.dat', '_inverted.dat')
            self.mapping_file = self.data_file.replace('.dat', '_mapping.dat')
            self.histograms_file = None

        self.codebook_file = codebook_file
        self.codebook = None
        if codebook_file and os.path.exists(codebook_file):
            self.codebook = Codebook(k=vocabulary_size, codebook_file=codebook_file)
            self.codebook.load_from_disk()

        self.extractor = extractor

        self.total_docs = 0
        self.id_size = 16
        self.path_size = 240
        self.fixed_path_size = 256
        self._load_mapping()

    # Se cargan datos como la cantidad de documentos, el tamaño del ID y del path
    def _load_mapping(self):
        if not os.path.exists(self.mapping_file):
            self.total_docs = 0
            self.id_size = 16
            self.path_size = 240
            return

        with open(self.mapping_file, 'rb') as f:
            header = f.read(12)
            if len(header) < 12:
                self.total_docs = 0
                self.id_size = 16
                self.path_size = 240
                return

            self.total_docs, self.id_size, self.path_size = struct.unpack('iii', header)
            self.fixed_path_size = self.id_size + self.path_size

    # Este recibe un Id y busca en los histogramas pre-calculados
    def _read_histogram(self, doc_id: int) -> np.ndarray:
        if not self.histograms_file or not os.path.exists(self.histograms_file):
            return None

        histogram_size_bytes = self.vocabulary_size * 4
        offset = 8 + (doc_id * histogram_size_bytes)

        with open(self.histograms_file, 'rb') as f:
            f.seek(offset)
            histogram = np.frombuffer(f.read(histogram_size_bytes), dtype=np.float32).copy()

        return histogram

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

    def _knnSearch_with_inverted(self, query_vector: np.ndarray, k: int) -> List[Tuple[Dict[str, Any], float]]:
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)

        max_heap = []

        # ESTRATEGIA: Índice invertido para filtrar candidatos relevantes
        MAX_WORDS = 35
        MAX_CANDIDATES = 10000

        active_indices = np.where(query_vector > 0)[0]
        if len(active_indices) == 0:
            return []

        weights = query_vector[active_indices]
        sorted_indices = active_indices[np.argsort(-weights)]
        top_codewords = sorted_indices[:MAX_WORDS].tolist()

        doc_scores = {}

        if not os.path.exists(self.inverted_index_file):
            return []

        with open(self.inverted_index_file, 'rb') as f:
            num_words = struct.unpack('i', f.read(4))[0]

            word_map = {}
            for _ in range(num_words):
                word_id = struct.unpack('i', f.read(4))[0]
                count = struct.unpack('i', f.read(4))[0]
                word_map[word_id] = (f.tell(), count)
                f.seek(count * 8, 1)

            for word_id in top_codewords:
                if word_id not in word_map:
                    continue

                offset, count = word_map[word_id]
                f.seek(offset)
                query_weight = query_vector[word_id]

                for _ in range(count):
                    doc_id, tfidf_weight = struct.unpack('if', f.read(8))
                    score = query_weight * tfidf_weight
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score

        if len(doc_scores) > MAX_CANDIDATES:
            top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:MAX_CANDIDATES]
            candidate_doc_ids = set(doc_id for doc_id, _ in top_docs)
        else:
            candidate_doc_ids = set(doc_scores.keys())

        sorted_candidates = sorted(candidate_doc_ids)

        for doc_id in sorted_candidates:
            doc_vector = self._read_histogram(doc_id)
            if doc_vector is None:
                continue

            distance = self._cosine_distance(query_vector, doc_vector)

            if len(max_heap) < k:
                heapq.heappush(max_heap, (-distance, doc_id, doc_vector))
            elif distance < -max_heap[0][0]:
                heapq.heapreplace(max_heap, (-distance, doc_id, doc_vector))

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
        if not self.extractor:
            raise RuntimeError("Extractor no configurado")

        if not self.codebook:
            raise RuntimeError("Codebook no está inicializado")

        descriptors = self.extractor.extract(file_path)
        query_vector = self.codebook.descriptors_to_histogram(descriptors, apply_tfidf=True)

        return self._knnSearch_with_inverted(query_vector, k)

    def _cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 1.0

        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(1.0 - similarity)

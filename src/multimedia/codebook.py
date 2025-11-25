import numpy as np
import struct
import os
import sys
from typing import List
from sklearn.cluster import MiniBatchKMeans

class Codebook:
    def __init__(self, k: int = 1000, descriptor_type: str = "IMAGE", codebook_file: str = None):
        self.k = k
        self.descriptor_type = descriptor_type
        self.codebook_file = codebook_file
        self.codewords = None
        self.kmeans = None
        self.dim = None
        self.idf_values = None
        self.N = 0

    def build(self, all_descriptors: List[np.ndarray], batch_size: int = 10000):
        valid_descriptors = []

        for desc in all_descriptors:
            if desc.size == 0:
                continue
            valid_descriptors.append(desc)

        if not valid_descriptors:
            raise RuntimeError("No se extrajeron descriptores")

        all_desc = np.vstack(valid_descriptors).astype(np.float32)
        self.dim = all_desc.shape[1]

        print(f"  K-means: {len(all_desc):,} descriptores, k={self.k}")
        sys.stdout.flush()

        self.kmeans = MiniBatchKMeans(
            n_clusters=self.k,
            batch_size=batch_size,
            random_state=42,
            verbose=1,
            max_iter=2000
        )
        self.kmeans.fit(all_desc)

        self.codewords = self.kmeans.cluster_centers_.astype(np.float32)
        self.dim = self.codewords.shape[1]

        print(f"Codebook listo (k={self.k}, dim={self.dim})\n")
        sys.stdout.flush()

    def compute_idf(self, all_histograms: List[np.ndarray], num_docs: int):
        self.N = num_docs
        df = np.zeros(self.k, dtype=np.float32)

        for histogram in all_histograms:
            present_words = histogram > 0
            df += present_words.astype(np.float32)

        self.idf_values = np.log((num_docs + 1) / (df + 1)) + 1
        self.idf_values = self.idf_values.astype(np.float32)

        print(f"IDF calculado: {np.min(self.idf_values):.3f} - {np.max(self.idf_values):.3f}")
        sys.stdout.flush()

    def descriptors_to_histogram(self, descriptors: np.ndarray, apply_tfidf: bool = False) -> np.ndarray:
        if descriptors.size == 0:
            return np.zeros(self.k, dtype=np.float32)

        descriptors = descriptors.astype(np.float32)

        if self.kmeans is None:
            self._read_from_disk()

        labels = self.kmeans.predict(descriptors)
        histogram = np.bincount(labels, minlength=self.k).astype(np.float32)

        if apply_tfidf:
            histogram = self.apply_tfidf_to_histogram(histogram)

        return histogram

    def apply_tfidf_to_histogram(self, histogram: np.ndarray) -> np.ndarray:
        total_words = histogram.sum()
        if total_words > 0:
            histogram = histogram / total_words

        histogram = histogram * self.idf_values

        norm = np.linalg.norm(histogram)
        if norm > 0:
            histogram = histogram / norm

        return histogram

    def save(self, filepath: str):
        if self.codewords is None:
            raise RuntimeError("Codebook no entrenado")

        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(filepath, 'wb') as f:
            header = struct.pack('ii', self.k, self.dim)
            f.write(header)

            codewords_bytes = self.codewords.astype(np.float32).tobytes()
            f.write(codewords_bytes)

            idf_bytes = self.idf_values.astype(np.float32).tobytes()
            f.write(idf_bytes)
        self.codewords = None
        self.kmeans = None
        self.codebook_file = filepath

    def _read_from_disk(self):
        if not self.codebook_file or not os.path.exists(self.codebook_file):
            raise RuntimeError(f"Codebook no encontrado: {self.codebook_file}")

        with open(self.codebook_file, 'rb') as f:
            header_bytes = f.read(8)
            k, dim = struct.unpack('ii', header_bytes)

            self.k = k
            self.dim = dim

            num_floats = k * dim
            codewords_bytes = f.read(num_floats * 4)
            codewords = np.frombuffer(codewords_bytes, dtype=np.float32).reshape(k, dim)

            idf_bytes = f.read(k * 4)
            self.idf_values = np.frombuffer(idf_bytes, dtype=np.float32)

        self.kmeans = MiniBatchKMeans(n_clusters=self.k, random_state=42)
        self.kmeans.cluster_centers_ = codewords

        self.kmeans._n_threads = 1
        self.kmeans._n_features_in = dim

        return codewords

    def load_from_disk(self):
        self.codewords = self._read_from_disk()

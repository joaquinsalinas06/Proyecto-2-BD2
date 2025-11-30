"""
KNN Inverted - Búsqueda usando índice invertido con filtro de candidatos.
Implementa filtro para limitar el número de comparaciones usando palabras más discriminativas.
"""
import struct
import numpy as np
from typing import List, Tuple, Dict


class KNNInverted:
    """
    Índice KNN Invertido con filtro para limitar candidatos.
    
    El filtro selecciona solo las palabras visuales más discriminativas del query
    (aquellas con mayor TF-IDF) para limitar el número de candidatos a comparar.
    Esto evita el problema de retornar 35k de 40k documentos cuando las palabras
    más comunes son muy frecuentes.
    """
    
    def __init__(self, inverted_file: str, histograms_file: str, mapping_file: str,
                 max_candidates: int = 10000):
        """
        Inicializa el índice KNN invertido con filtro.
        
        Args:
            inverted_file: Ruta al archivo del índice invertido
            histograms_file: Ruta al archivo de histogramas TF-IDF
            mapping_file: Ruta al archivo de mapeo doc_id -> product_id
            max_candidates: Número máximo de candidatos a considerar (default: 10000)
        """
        self.inverted_file = inverted_file
        self.histograms_file = histograms_file
        self.mapping_file = mapping_file
        self.max_candidates = max_candidates
        
        self.inverted_index = None
        self.histograms = None
        self.mapping = None
        self.num_docs = 0
        self.vocab_size = 0
    
    def load_index(self):
        """Carga el índice desde disco."""
        # Cargar índice invertido
        self.inverted_index = {}
        
        with open(self.inverted_file, 'rb') as f:
            data = f.read()
        
        offset = 0
        num_words = struct.unpack_from('i', data, offset)[0]
        offset += 4
        
        for _ in range(num_words):
            word_id = struct.unpack_from('i', data, offset)[0]
            offset += 4
            
            num_docs = struct.unpack_from('i', data, offset)[0]
            offset += 4
            
            docs = []
            if num_docs > 0:
                for _ in range(num_docs):
                    doc_id, weight = struct.unpack_from('if', data, offset)
                    docs.append((doc_id, weight))
                    offset += 8
                
                self.inverted_index[word_id] = docs
        
        # Cargar histogramas
        with open(self.histograms_file, 'rb') as f:
            self.num_docs, self.vocab_size = struct.unpack('ii', f.read(8))
            
            histograms = []
            for _ in range(self.num_docs):
                hist_bytes = f.read(self.vocab_size * 4)
                hist = np.frombuffer(hist_bytes, dtype=np.float32)
                histograms.append(hist)
            
            self.histograms = np.array(histograms)
        
        # Cargar mapping
        mapping = {}
        with open(self.mapping_file, 'rb') as f:
            num_docs, id_size, path_size = struct.unpack('iii', f.read(12))
            
            for doc_id in range(num_docs):
                id_bytes = f.read(id_size)
                path_bytes = f.read(path_size)
                
                product_id = id_bytes.rstrip(b'\x00').decode('utf-8')
                path = path_bytes.rstrip(b'\x00').decode('utf-8')
                
                mapping[doc_id] = (product_id, path)
        
        self.mapping = mapping
    
    def _select_top_query_words(self, query_vector: np.ndarray, 
                                target_candidates: int) -> List[int]:
        """
        Selecciona las palabras del query más discriminativas (mayor TF-IDF).
        
        Itera agregando palabras en orden de TF-IDF descendente hasta que
        el número estimado de candidatos alcance el target.
        
        Args:
            query_vector: Vector de características del query
            target_candidates: Número objetivo de candidatos
            
        Returns:
            Lista de word_ids seleccionados
        """
        # Obtener palabras activas en el query con sus pesos TF-IDF
        active_words = []
        for word_id in np.nonzero(query_vector)[0]:
            weight = query_vector[word_id]
            if word_id in self.inverted_index:
                # Número de docs que contienen esta palabra
                num_docs_with_word = len(self.inverted_index[word_id])
                active_words.append((word_id, weight, num_docs_with_word))
        
        if not active_words:
            return []
        
        # Ordenar por TF-IDF descendente (mayor peso = más discriminativo)
        active_words.sort(key=lambda x: x[1], reverse=True)
        
        # Seleccionar palabras hasta alcanzar el target de candidatos
        selected_words = []
        candidates_set = set()
        
        for word_id, weight, num_docs in active_words:
            # Agregar candidatos de esta palabra
            for doc_id, _ in self.inverted_index[word_id]:
                candidates_set.add(doc_id)
            
            selected_words.append(word_id)
            
            # Si ya alcanzamos el target, detenerse
            if len(candidates_set) >= target_candidates:
                break
        
        return selected_words
    
    def search(self, query_vector: np.ndarray, k: int = 8) -> List[Tuple[int, float]]:
        """
        Realiza búsqueda KNN usando índice invertido con filtro.
        
        El filtro limita el número de candidatos considerando solo las palabras
        más discriminativas del query (aquellas con mayor TF-IDF).
        
        Args:
            query_vector: Vector de características de la query
            k: Número de vecinos más cercanos a retornar
            
        Returns:
            Lista de tuplas (doc_id, similitud) ordenadas por similitud descendente
        """
        if self.inverted_index is None or self.histograms is None:
            raise RuntimeError("Debe llamar a load_index() antes de realizar búsquedas")
        
        # Aplicar filtro: seleccionar solo palabras más discriminativas
        selected_words = self._select_top_query_words(query_vector, self.max_candidates)
        
        # Calcular scores solo para documentos con palabras seleccionadas
        scores_dict = {}
        query_norm_sq = np.sum(query_vector ** 2)
        
        if query_norm_sq == 0:
            return []
        
        query_norm = np.sqrt(query_norm_sq)
        
        # Acumular scores solo para palabras seleccionadas
        for word_id in selected_words:
            if word_id in self.inverted_index:
                query_weight = query_vector[word_id]
                for doc_id, doc_weight in self.inverted_index[word_id]:
                    if doc_id not in scores_dict:
                        scores_dict[doc_id] = 0.0
                    scores_dict[doc_id] += query_weight * doc_weight
        
        # Normalizar por la norma del query
        for doc_id in scores_dict:
            scores_dict[doc_id] /= query_norm
        
        # Obtener top-k
        if len(scores_dict) == 0:
            return []
        
        sorted_docs = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_docs[:k]
        
        results = [(int(doc_id), float(score)) for doc_id, score in top_k]
        return results
    
    def get_vector(self, doc_id: int) -> np.ndarray:
        """Retorna el vector de características de un documento."""
        if self.histograms is None:
            raise RuntimeError("Debe llamar a load_index() antes de acceder a vectores")
        return self.histograms[doc_id]
    
    def get_product_id(self, doc_id: int) -> str:
        """Retorna el product_id de un documento."""
        if self.mapping is None:
            raise RuntimeError("Debe llamar a load_index() antes de acceder al mapping")
        product_id, _ = self.mapping[doc_id]
        return product_id
    
    def get_num_docs(self) -> int:
        """Retorna el número de documentos indexados."""
        return self.num_docs
    
    def get_vocab_size(self) -> int:
        """Retorna el tamaño del vocabulario."""
        return self.vocab_size
    
    def get_num_candidates_for_query(self, query_vector: np.ndarray) -> int:
        """
        Retorna el número estimado de candidatos que se considerarían
        para un query dado con el filtro actual.
        """
        selected_words = self._select_top_query_words(query_vector, self.max_candidates)
        
        candidates_set = set()
        for word_id in selected_words:
            if word_id in self.inverted_index:
                for doc_id, _ in self.inverted_index[word_id]:
                    candidates_set.add(doc_id)
        
        return len(candidates_set)

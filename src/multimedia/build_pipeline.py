import os
import glob
import struct
import numpy as np
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count
from .codebook import Codebook
from .feature_extractors import SIFTExtractor
from .feature_extractors import MFCCExtractor

_global_extractor = None
_global_extractor_type = None

def _init_worker(extractor_type):
    global _global_extractor, _global_extractor_type
    _global_extractor_type = extractor_type

    if extractor_type == 'SIFT':
        _global_extractor = SIFTExtractor()
    elif extractor_type == 'MFCC':
        _global_extractor = MFCCExtractor()
    else:
        raise ValueError(f"Tipo de extractor no soportado: {extractor_type}")

def _extract_batch(paths_batch):
    global _global_extractor
    results = []
    for path in paths_batch:
        try:
            desc = _global_extractor.extract(path)
            if len(desc.shape) == 1:
                desc = desc.reshape(1, -1)
            results.append((path, desc))
        except:
            pass
    return results

def extract_features_parallel(file_paths: List[str], extractor) -> Dict[str, np.ndarray]:
    num_workers = cpu_count()
    batch_size = max(1, len(file_paths) // (num_workers * 10))
    batches = [file_paths[i:i+batch_size] for i in range(0, len(file_paths), batch_size)]

    extractor_class_name = extractor.__class__.__name__
    if 'SIFT' in extractor_class_name:
        extractor_type = 'SIFT'
    elif 'MFCC' in extractor_class_name:
        extractor_type = 'MFCC'
    else:
        extractor_type = 'SIFT'

    features_map = {}

    with Pool(processes=num_workers, initializer=_init_worker, initargs=(extractor_type,)) as pool:
        for batch_results in pool.imap_unordered(_extract_batch, batches):
            for path, desc in batch_results:
                features_map[path] = desc

    return features_map

def build_knn_index_from_collection(
    collection_path: str,
    codebook_file: str,
    vocabulary_size: int,
    extractor,
    inverted_file: str,
    mapping_file: str,
    histograms_file: str = None,
    buckets_file: str = None,
    verbose: bool = True,
    id_extractor = None
) -> int:
    print(f"Construyendo dataset desde {collection_path}...")

    patterns = ['*.jpg', '*.jpeg', '*.png', '*.wav', '*.mp3']
    file_paths = []
    for pattern in patterns:
        file_paths.extend(glob.glob(os.path.join(collection_path, pattern)))

    if not file_paths:
        raise ValueError(f"No se encontraron archivos en {collection_path}")

    print(f"Encontrados {len(file_paths)} archivos")

    features_map = extract_features_parallel(file_paths, extractor)

    print(f"Extraídos {len(features_map)} conjuntos de descriptores")

    all_descriptors = list(features_map.values())
    file_paths_order = list(features_map.keys())

    codebook = Codebook(k=vocabulary_size, descriptor_type="IMAGE", codebook_file=codebook_file)
  
    if os.path.exists(codebook_file):
        print(f"Cargando codebook existente: {codebook_file}")
        codebook.load_from_disk()
    else:
        print("Construyendo nuevo codebook...")
        codebook.build(all_descriptors, batch_size=10000)

    print("Calculando histogramas TF...")
    all_histograms_tf = []
    for descriptors in all_descriptors:
        histogram_tf = codebook.descriptors_to_histogram(descriptors, apply_tfidf=False)
        all_histograms_tf.append(histogram_tf)

    codebook.compute_idf(all_histograms_tf, num_docs=len(all_descriptors))
    codebook.save(codebook_file)

    print(f"Codebook guardado: {codebook_file}")

    print("Construyendo índice invertido TF-IDF...")

    inverted_index = {}
    doc_id_to_path = {}

    for i, histogram_tf in enumerate(all_histograms_tf):
        histogram_tfidf = codebook.apply_tfidf_to_histogram(histogram_tf)
        doc_id_to_path[i] = file_paths_order[i]

        for word_id, value in enumerate(histogram_tfidf):
            if value > 0:
                if word_id not in inverted_index:
                    inverted_index[word_id] = []
                inverted_index[word_id].append((i, float(value)))

    print(f"Índice invertido: {len(inverted_index)} palabras")

    print(f"Guardando mapping: {mapping_file}")
    
    ID_SIZE = 16
    PATH_SIZE = 240
    ENTRY_SIZE = ID_SIZE + PATH_SIZE

    with open(mapping_file, 'wb') as f:
        f.write(struct.pack('iii', len(doc_id_to_path), ID_SIZE, PATH_SIZE))

        for doc_id in sorted(doc_id_to_path.keys()):
            path = doc_id_to_path[doc_id]

            if id_extractor:
                try:
                    product_id = str(id_extractor(path))
                except:
                    product_id = str(doc_id)
            else:
                product_id = str(doc_id)
            
            if len(product_id) > ID_SIZE:
                product_id = product_id[:ID_SIZE]
            id_bytes = product_id.encode('utf-8').ljust(ID_SIZE, b'\x00')
            
            if len(path) > PATH_SIZE:
                path = path[:PATH_SIZE]
            path_bytes = path.encode('utf-8').ljust(PATH_SIZE, b'\x00')
            
            f.write(id_bytes)
            f.write(path_bytes)

    print(f"Guardando índice invertido: {inverted_file}")

    with open(inverted_file, 'wb') as f:
        f.write(struct.pack('i', len(inverted_index)))
        sorted_items = sorted(inverted_index.items())
        for word_id, docs_w in sorted_items:
            f.write(struct.pack('i', word_id))
            f.write(struct.pack('i', len(docs_w)))
            for doc_id, weight in docs_w:
                f.write(struct.pack('if', doc_id, weight))

    if histograms_file:
        print(f"Guardando histogramas: {histograms_file}")

        with open(histograms_file, 'wb') as f:
            f.write(struct.pack('ii', len(all_histograms_tf), vocabulary_size))

            for histogram_tf in all_histograms_tf:
                histogram_tfidf = codebook.apply_tfidf_to_histogram(histogram_tf)
                f.write(histogram_tfidf.astype(np.float32).tobytes())

    
    if buckets_file:
        print(f"Guardando buckets: {buckets_file}")

        all_histograms_tfidf = []
        for histogram_tf in all_histograms_tf:
            histogram_tfidf = codebook.apply_tfidf_to_histogram(histogram_tf)
            all_histograms_tfidf.append(histogram_tfidf.astype(np.float32))

        buckets_data = {}

        for doc_id, histogram_tfidf in enumerate(all_histograms_tfidf):
            active_codewords = np.where(histogram_tfidf > 0)[0]

            for word_id in active_codewords:
                if word_id not in buckets_data:
                    buckets_data[word_id] = []
                buckets_data[word_id].append(doc_id)

        entry_size = 4
        header_size = 8
        index_size = vocabulary_size * 12

        bucket_offsets = []
        current_offset = header_size + index_size

        for word_id in range(vocabulary_size):
            if word_id in buckets_data:
                count = len(buckets_data[word_id])
            else:
                count = 0
            bucket_offsets.append((current_offset, count))
            current_offset += count * entry_size

        with open(buckets_file, 'wb') as f:
            f.write(struct.pack('ii', vocabulary_size, vocabulary_size))

            for offset, count in bucket_offsets:
                f.write(struct.pack('qi', offset, count))

            for word_id in range(vocabulary_size):
                if word_id in buckets_data:
                    for doc_id in buckets_data[word_id]:
                        f.write(struct.pack('i', doc_id))

        total_entries = sum(len(docs) for docs in buckets_data.values())
        avg_bucket_size = total_entries / vocabulary_size if vocabulary_size > 0 else 0
        max_bucket_size = max(len(docs) for docs in buckets_data.values()) if buckets_data else 0

        print(f"   Buckets: {len(buckets_data)}/{vocabulary_size} no vacíos")
        print(f"   Entries totales: {total_entries} (promedio {avg_bucket_size:.1f}/bucket)")
        print(f"   Bucket más grande: {max_bucket_size} docs")
        print(f"   Tamaño archivo: {current_offset / 1024 / 1024:.1f} MB")

    if verbose:
        print(f"Dataset construido: {len(features_map)} archivos indexados")

    return len(features_map)

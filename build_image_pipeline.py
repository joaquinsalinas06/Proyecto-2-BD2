import os
import sys
import glob
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from src.multimedia.feature_extractors import SIFTExtractor
from src.multimedia.codebook import Codebook
import struct

k_val = 2000

FASHION_DIR = "data/fashion-dataset/fashion-dataset/images"
FEATURES_FILE = "data/features_sift.dat"

def _extract_features_batch(paths_batch):
    extractor = SIFTExtractor()
    results = []
    for path in paths_batch:
        try:
            desc = extractor.extract(path)
            if len(desc.shape) == 1:
                desc = desc.reshape(1, -1)
            results.append((path, desc))
        except Exception:
            results.append((path, None))
    return results

def save_features_to_dat(file_paths, dat_file):
    num_workers = cpu_count()

    features_map = {}
    processed = 0

    batch_size = len(file_paths) // (num_workers * 10)
    if batch_size < 10:
        batch_size = 10

    batches = []
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        batches.append(batch)


    # Consultar Profe
    pool = Pool(processes=num_workers)
    for batch_results in pool.imap_unordered(_extract_features_batch, batches):
        for path, desc in batch_results:
            if desc is not None:
                features_map[path] = desc
                processed += 1

        if processed % 1000 == 0:
            if processed > 0:
                pct = (processed / len(file_paths)) * 100
                print(f"  {processed}/{len(file_paths)} ({pct:.1f}%)")

    pool.close()
    pool.join()

    print(f"Guardando {len(features_map)} features")
    with open(dat_file, 'wb') as f:
        n_images = len(features_map)
        header_bytes = struct.pack('i', n_images)
        f.write(header_bytes)

        for path, descriptors in features_map.items():
            path_bytes = path.encode('utf-8')
            path_len = len(path_bytes)
            path_len_bytes = struct.pack('i', path_len)
            f.write(path_len_bytes)
            f.write(path_bytes)

            n_desc, dim = descriptors.shape
            shape_bytes = struct.pack('ii', n_desc, dim)
            f.write(shape_bytes)

            desc_bytes = descriptors.astype(np.float32).tobytes()
            f.write(desc_bytes)

    return processed

def load_features_from_dat(dat_file):
    features_map = {}

    with open(dat_file, 'rb') as f:
        header_bytes = f.read(4)
        n_images = struct.unpack('i', header_bytes)[0]

        for _ in range(n_images):
            path_len_bytes = f.read(4)
            path_len = struct.unpack('i', path_len_bytes)[0]
            path_bytes = f.read(path_len)
            path = path_bytes.decode('utf-8')

            shape_bytes = f.read(8)
            n_desc, dim = struct.unpack('ii', shape_bytes)

            num_floats = n_desc * dim
            desc_bytes = f.read(num_floats * 4)
            descriptors = np.frombuffer(desc_bytes, dtype=np.float32)
            descriptors = descriptors.reshape(n_desc, dim)

            features_map[path] = descriptors

    return features_map

def main():
    if not os.path.exists(FASHION_DIR):
        print(f"ERROR: No se encontro {FASHION_DIR}")
        sys.exit(1)

    os.makedirs("indices", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    t_total_start = time.time()

    print("Extraccion de Features SIFT (Si es necesario)")

    if os.path.exists(FEATURES_FILE):
        print(f"Cargando features existentes: {FEATURES_FILE}")
        features_map = load_features_from_dat(FEATURES_FILE)
        print(f"Total: {len(features_map)} imagenes")
    else:
        image_files = glob.glob(f"{FASHION_DIR}/*.jpg")
        if not image_files:
            print("ERROR: No se encontraron imagenes")
            sys.exit(1)

        print("Extrayendo features SIFT...")
        t0 = time.time()

        processed = save_features_to_dat(image_files, FEATURES_FILE)

        tiempo = time.time() - t0
        print(f"Extraccion completada: {processed} imagenes en {tiempo/60:.1f} min")

        features_map = load_features_from_dat(FEATURES_FILE)

    print("Construccion de Codebook")

    codebook_file = f"indices/fashion_k{k_val}_codebook.dat"

    all_descriptors = []
    file_paths_order = []
    for file_path, descriptors in features_map.items():
        all_descriptors.append(descriptors)
        file_paths_order.append(file_path)

    print(f"Total: {len(all_descriptors)} imagenes")

    codebook = Codebook(k=k_val, descriptor_type="IMAGE", codebook_file=codebook_file)

    codebook.build(all_descriptors, batch_size=10000)

    print("Primera pasada: calculando histogramas Frecuencia de Terminos")
    all_histograms_tf = []

    for i, descriptors in enumerate(all_descriptors):
        histogram_tf = codebook.descriptors_to_histogram(descriptors, apply_tfidf=False)
        all_histograms_tf.append(histogram_tf)

        count = i + 1
        if count % 5000 == 0:
            pct = (count / len(all_descriptors)) * 100
            print(f"  {count}/{len(all_descriptors)} ({pct:.1f}%)")

    print("Calculando IDF")
    codebook.compute_idf(all_histograms_tf, N=len(all_descriptors))

    print(f"Guardando codebook: {codebook_file}")
    codebook.save(codebook_file)

    print("PASO 3: Construccion de Indice Invertido TF-IDF")

    print("Aplicando TF-IDF a histogramas ya calculados...")
    inverted_index = {}
    num_processed = 0

    for i, histogram_tf in enumerate(all_histograms_tf):
        histogram_tfidf = codebook.apply_tfidf_to_histogram(histogram_tf)

        num_nonzero = np.count_nonzero(histogram_tfidf)
        if i < 200:
            print(f"  Histograma {i}: {num_nonzero}/{len(histogram_tfidf)} valores no-cero")
            print(f"    Min: {np.min(histogram_tfidf):.4f}, Max: {np.max(histogram_tfidf):.4f}, Mean: {np.mean(histogram_tfidf):.4f}")

        for word_id, value in enumerate(histogram_tfidf):
            if value > 0:
                if word_id not in inverted_index:
                    inverted_index[word_id] = []
                file_path = file_paths_order[i]
                weight = float(value)
                inverted_index[word_id].append((file_path, weight))

        num_processed += 1
        if num_processed % 5000 == 0:
            pct = (num_processed / len(all_histograms_tf)) * 100
            print(f"  {num_processed}/{len(all_histograms_tf)} ({pct:.1f}%)")

    print(f"Listo: {num_processed} histogramas TF-IDF")
    print(f"Indice invertido: {len(inverted_index)} palabras")

    index_file = f"indices/fashion_k{k_val}_index.dat"
    inverted_file = index_file.replace('.dat', '_inverted.dat')

    print(f"Guardando indice invertido: {inverted_file}")
    with open(inverted_file, 'wb') as f:
        num_words = len(inverted_index)
        num_words_bytes = struct.pack('i', num_words)
        f.write(num_words_bytes)

        sorted_items = sorted(inverted_index.items())
        for word_id, imgs_w in sorted_items:
            word_id_bytes = struct.pack('i', word_id)
            f.write(word_id_bytes)

            num_imagenes = len(imgs_w)
            num_imagenes_bytes = struct.pack('i', num_imagenes)
            f.write(num_imagenes_bytes)

            for file_path, weight in imgs_w:
                path_bytes = file_path.encode('utf-8')
                path_len = len(path_bytes)
                path_len_bytes = struct.pack('i', path_len)
                f.write(path_len_bytes)
                f.write(path_bytes)

                weight_bytes = struct.pack('f', weight)
                f.write(weight_bytes)

    t_total = time.time() - t_total_start
    print(f"Completado en {t_total/60:.1f} min")

if __name__ == "__main__":
    main()

import os
import glob
import shutil

from src.multimedia.feature_extractors import SIFTExtractor
from src.multimedia.build_pipeline import build_knn_index_from_collection

def extract_product_id_from_path(path: str) -> str:
    filename = os.path.basename(path)
    return os.path.splitext(filename)[0]

if __name__ == '__main__':
    import sys
    
    # Si se pasa un argumento, solo procesar ese tamaño
    # Ejemplo: python setup_fashion_indices.py 1000
    if len(sys.argv) > 1:
        try:
            vals = [int(arg) for arg in sys.argv[1:]]
        except ValueError:
            print("Error: Los argumentos deben ser números")
            print("Uso: python setup_fashion_indices.py [tamaño1] [tamaño2] ...")
            print("Ejemplo: python setup_fashion_indices.py 1000")
            print("Ejemplo: python setup_fashion_indices.py 1000 2000 4000")
            sys.exit(1)
    else:
        # Sin argumentos, procesar solo 1000 por defecto
        vals = [1000]

    print(f"Procesando tamaños: {vals}")
    print("="*60)

    for v in vals:
        SAMPLE_SIZE = v

        DATA_DIR = "Fashion/images"
        INDICES_DIR = f"data/indices/fashion/fashion_{SAMPLE_SIZE}"

        print("Construyendo indice para muestra de tamaño:", SAMPLE_SIZE)

        subset_dir = os.path.join(DATA_DIR, f"subset_{SAMPLE_SIZE}")
        os.makedirs(subset_dir, exist_ok=True)

        all_images = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")))[:SAMPLE_SIZE]

        print(f"Creando subconjunto: {len(all_images)} imágenes")
        for img in all_images:
            dst = os.path.join(subset_dir, os.path.basename(img))
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(img), dst)
                except OSError:
                    shutil.copy2(img, dst)

        os.makedirs(INDICES_DIR, exist_ok=True)

        prefix = f"fashion_{SAMPLE_SIZE}"
        codebook_file = os.path.join(INDICES_DIR, f"{prefix}_codebook.dat")
        inverted_file = os.path.join(INDICES_DIR, f"{prefix}_inverted.dat")
        mapping_file = os.path.join(INDICES_DIR, f"{prefix}_mapping.dat")
        histograms_file = os.path.join(INDICES_DIR, f"{prefix}_histograms.dat")
        buckets_file = os.path.join(INDICES_DIR, f"{prefix}_buckets.dat")

        extractor = SIFTExtractor()
        vocabulary_size = 2000  # Tamaño del vocabulario visual (palabras visuales)

        num_indexed = build_knn_index_from_collection(
            collection_path=subset_dir,
            codebook_file=codebook_file,
            vocabulary_size=vocabulary_size,
            extractor=extractor,
            inverted_file=inverted_file,
            mapping_file=mapping_file,
            histograms_file=histograms_file,
            buckets_file=buckets_file,
            verbose=True,
            id_extractor=extract_product_id_from_path
        )

        print(f"Índice construido con {num_indexed} imágenes para muestra de tamaño {SAMPLE_SIZE}\n")

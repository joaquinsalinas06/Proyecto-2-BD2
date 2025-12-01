"""
Script de prueba para verificar que las clases KNN funcionan correctamente.
Este script realiza una búsqueda de prueba con ambos métodos.

Usa las clases wrapper (KNNSequential y KNNInverted) que internamente:
- KNNSequential -> KNNSequentialIndex.knnSearch (búsqueda secuencial completa)
- KNNInverted -> KNNInvertedIndex._knnSearch_with_inverted (con filtrado inteligente)
"""
import os
import sys
import numpy as np

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.multimedia.knn_wrapper import KNNSequential, KNNInverted


def test_knn_sequential(sample_size=1000):
    """Prueba la clase KNNSequential."""
    print("="*60)
    print("Probando KNNSequential")
    print("="*60)
    
    indices_dir = f"data/indices/fashion/fashion_{sample_size}"
    histograms_file = os.path.join(indices_dir, f"fashion_{sample_size}_histograms.dat")
    mapping_file = os.path.join(indices_dir, f"fashion_{sample_size}_mapping.dat")
    
    if not os.path.exists(histograms_file):
        print(f"Error: No se encontró {histograms_file}")
        return False
    
    try:
        # Crear y cargar índice
        knn = KNNSequential(histograms_file, mapping_file)
        knn.load_index()
        
        print(f"✓ Índice cargado correctamente")
        print(f"  - Documentos: {knn.get_num_docs()}")
        print(f"  - Vocabulario: {knn.get_vocab_size()}")
        
        # Realizar una búsqueda de prueba
        query_idx = 0
        query_vector = knn.get_vector(query_idx)
        results = knn.search(query_vector, k=5)
        
        print(f"\n✓ Búsqueda exitosa")
        print(f"  Query: {knn.get_product_id(query_idx)}")
        print(f"  Top-5 resultados:")
        for rank, (doc_id, score) in enumerate(results, 1):
            print(f"    {rank}. {knn.get_product_id(doc_id)} (score: {score:.4f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_knn_inverted(sample_size=1000, max_candidates=10000):
    """Prueba la clase KNNInverted."""
    print("\n" + "="*60)
    print("Probando KNNInverted")
    print("="*60)
    
    indices_dir = f"data/indices/fashion/fashion_{sample_size}"
    inverted_file = os.path.join(indices_dir, f"fashion_{sample_size}_inverted.dat")
    histograms_file = os.path.join(indices_dir, f"fashion_{sample_size}_histograms.dat")
    mapping_file = os.path.join(indices_dir, f"fashion_{sample_size}_mapping.dat")
    
    if not os.path.exists(inverted_file):
        print(f"Error: No se encontró {inverted_file}")
        return False
    
    try:
        # Crear y cargar índice
        knn = KNNInverted(inverted_file, histograms_file, mapping_file, 
                         max_candidates=max_candidates)
        knn.load_index()
        
        print(f"✓ Índice cargado correctamente")
        print(f"  - Documentos: {knn.get_num_docs()}")
        print(f"  - Vocabulario: {knn.get_vocab_size()}")
        print(f"  - Límite candidatos: {max_candidates}")
        
        # Realizar una búsqueda de prueba
        query_idx = 0
        query_vector = knn.get_vector(query_idx)
        
        # Contar candidatos
        num_candidates = knn.get_num_candidates_for_query(query_vector)
        print(f"\n✓ Análisis de query")
        print(f"  Candidatos que se considerarán: {num_candidates}")
        print(f"  Porcentaje del total: {num_candidates / knn.get_num_docs() * 100:.1f}%")
        
        # Realizar búsqueda
        results = knn.search(query_vector, k=5)
        
        print(f"\n✓ Búsqueda exitosa")
        print(f"  Query: {knn.get_product_id(query_idx)}")
        print(f"  Top-5 resultados:")
        for rank, (doc_id, score) in enumerate(results, 1):
            print(f"    {rank}. {knn.get_product_id(doc_id)} (score: {score:.4f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_results(sample_size=1000):
    """Compara los resultados de ambos métodos."""
    print("\n" + "="*60)
    print("Comparando resultados de ambos métodos")
    print("="*60)
    
    try:
        # Cargar ambos índices
        indices_dir = f"data/indices/fashion/fashion_{sample_size}"
        histograms_file = os.path.join(indices_dir, f"fashion_{sample_size}_histograms.dat")
        mapping_file = os.path.join(indices_dir, f"fashion_{sample_size}_mapping.dat")
        inverted_file = os.path.join(indices_dir, f"fashion_{sample_size}_inverted.dat")
        
        knn_seq = KNNSequential(histograms_file, mapping_file)
        knn_seq.load_index()
        
        knn_inv = KNNInverted(inverted_file, histograms_file, mapping_file, 
                             max_candidates=10000)
        knn_inv.load_index()
        
        # Probar con el mismo query
        query_idx = 0
        query_vector = knn_seq.get_vector(query_idx)
        
        results_seq = knn_seq.search(query_vector, k=5)
        results_inv = knn_inv.search(query_vector, k=5)
        
        print(f"Query: {knn_seq.get_product_id(query_idx)}")
        print(f"\nTop-5 Sequential:")
        for rank, (doc_id, score) in enumerate(results_seq, 1):
            print(f"  {rank}. {knn_seq.get_product_id(doc_id)} ({score:.4f})")
        
        print(f"\nTop-5 Inverted (con filtro):")
        for rank, (doc_id, score) in enumerate(results_inv, 1):
            print(f"  {rank}. {knn_inv.get_product_id(doc_id)} ({score:.4f})")
        
        # Calcular overlap
        ids_seq = [doc_id for doc_id, _ in results_seq]
        ids_inv = [doc_id for doc_id, _ in results_inv]
        overlap = len(set(ids_seq) & set(ids_inv))
        
        print(f"\n✓ Overlap en Top-5: {overlap}/5 ({overlap/5*100:.0f}%)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    sample_size = 1000
    
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
        except ValueError:
            print(f"Error: Tamaño inválido '{sys.argv[1]}'")
            sys.exit(1)
    
    print(f"Probando con muestra de tamaño: {sample_size}\n")
    
    success = True
    
    # Test Sequential
    if not test_knn_sequential(sample_size):
        success = False
    
    # Test Inverted
    if not test_knn_inverted(sample_size):
        success = False
    
    # Comparación
    if success:
        compare_results(sample_size)
    
    print("\n" + "="*60)
    if success:
        print("✓ Todas las pruebas pasaron correctamente")
    else:
        print("✗ Algunas pruebas fallaron")
    print("="*60)

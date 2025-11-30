#!/usr/bin/env python

import sys
import subprocess
import os


def run_command(cmd, description):
    print("\n" + "="*60)
    print(f"{description}")
    print("="*60)
    print(f"Comando: {' '.join(cmd)}\n")
    

    result = subprocess.run(cmd, check=True)
    print(f"\n {description} completado")
    return True


def main():
    sample_size = 1000
    max_candidates = 10000
    
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
        except ValueError:
            print(f"Error: Tamaño inválido '{sys.argv[1]}'")
            print("Uso: python run_all_benchmarks.py [tamaño] [max_candidatos]")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            max_candidates = int(sys.argv[2])
        except ValueError:
            print(f"Error: Max candidatos inválido '{sys.argv[2]}'")
            print("Uso: python run_all_benchmarks.py [tamaño] [max_candidatos]")
            sys.exit(1)
    
    print("="*60)
    print("BENCHMARKS KNN FASHION")
    print("="*60)
    print(f"Tamaño de muestra: {sample_size}")
    print(f"Límite de candidatos (Inverted): {max_candidates}")
    print("="*60)
    
    success = True
    
    # 1. Probar clases KNN
    if not run_command(
        [sys.executable, "test_knn_classes.py", str(sample_size)],
        "Probando clases KNN"
    ):
        success = False
        print("\n¡Advertencia! Las pruebas fallaron, pero continuando con benchmarks...")
    
    # 2. Benchmark Sequential
    if not run_command(
        [sys.executable, os.path.join("benchmarks", "benchmark_fashion_knn_sequential.py"), 
         str(sample_size)],
        "Benchmark KNN Sequential"
    ):
        success = False
    
    # 3. Benchmark Inverted
    if not run_command(
        [sys.executable, os.path.join("benchmarks", "benchmark_fashion_knn_inverted.py"), 
         str(sample_size), str(max_candidates)],
        "Benchmark KNN Inverted (con filtro)"
    ):
        success = False
    
    # Resumen final
    print("\n" + "="*60)
    if success:
        print("todos los benchmarks completados exitosamente")
    else:
        print("Algunos benchmarks fallaron")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

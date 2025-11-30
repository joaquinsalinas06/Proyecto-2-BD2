import sys
import os
import csv

file_path = "spotify_songs_cleaned.csv"
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.table_manager import TableManager

if __name__ == "__main__":
    print("TEST: Índice Invertido (TEX_INV)")
    print("=" * 60)

    fin = int(input("Ingrese cantidad de registros a leer: "))
    canciones = []

    print(f"\nLeyendo primeros {fin} registros del CSV...\n")

    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for i, row in enumerate(reader):
                if i >= fin:  # Detener cuando se alcance el límite
                    break
                
                # Extraer los datos (ajusta los nombres de columnas según tu CSV)
                id_val = i + 1
                artista = row.get('artist', '').replace("'", "''")  # Escapar comillas
                cancion = row.get('song', '').replace("'", "''")
                link = row.get('link', '').replace("'", "''")
                texto = row.get('lyrics', '').replace("'", "''")  # ← CAMBIO: era genero, debe ser texto
                
                # Agregar a la lista
                canciones.append((id_val, artista, cancion, link, texto))
        
        print(f"✓ Se cargaron {len(canciones)} canciones\n")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{file_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
        sys.exit(1)

    tm = TableManager()
    
    print("\n1. Creando tabla con índice BTREE en la clave primaria...")
    sql = """
    CREATE TABLE canciones (
        id INT KEY INDEX BTREE,
        artist VARCHAR[100],
        song VARCHAR[200],
        link VARCHAR[50],
        texto VARCHAR[500]
    );
    """
    tm.sql(sql)
    print("   ✓ Tabla creada\n")

    print("2. Insertando canciones...")
    
    # ← CAMBIO: iterar correctamente sobre la tupla de 5 elementos
    for id_val, artista, cancion, link, texto in canciones:
        query = f"INSERT INTO canciones VALUES ({id_val}, '{artista}', '{cancion}', '{link}', '{texto}');"
        tm.sql(query)
    
    print(f"   ✓ Insertadas {len(canciones)} canciones\n")
   
    print("3. Construyendo índice invertido en 'texto'...")
    # ← CAMBIO: BUILD en la columna 'texto', no 'song'
    tm.sql("BUILD TEX_INV ON canciones(texto);")
    print("   ✓ Índice TEX_INV construido\n")
    
    print("4. Buscando canciones similares a 'losing'...")
    # ← CAMBIO: query correcta con nombres de columnas correctos
    results = tm.sql("""
    SELECT artist, song, texto FROM canciones    
    WHERE texto @@ 'losing'   
    LIMIT 9;
    """)
    
    print(f"\n{'='*60}")
    print(f"Resultados ({len(results[0]['data'])} canciones):")
    print(f"{'='*60}\n")
    
    # ← CAMBIO: nombres de columnas correctos (artist, song, texto)
    for i, row in enumerate(results[0]['data'], 1):
        score = row.get('_score', 0)
        print(f"{i}. {row['artist']} - {row['song']}")
        print(f"   Texto: {row['texto'][:100]}...")
        print(f"   Score: {score:.4f}\n")
    
    print(f"{'='*60}")
    print("✓ Test completado exitosamente")
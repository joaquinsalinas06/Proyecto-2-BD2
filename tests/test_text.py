import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.table_manager import TableManager

if __name__ == "__main__":
    print("TEST: Índice Invertido (TEX_INV)")
    print("=" * 60)

    tm = TableManager()
    
    print("\n1. Creando tabla con índice BTREE en la clave primaria...")
    sql = """
    CREATE TABLE canciones (
        id INT KEY INDEX BTREE,
        artista VARCHAR[100],
        cancion VARCHAR[200],
        genero VARCHAR[50]
    );
    """
    tm.sql(sql)
    print("   ✓ Tabla creada\n")

    print("2. Insertando canciones...")
    canciones = [
        (1, "The Beatles", "Hey Jude", "Rock"),
        (2, "The Beatles", "Let It Be", "Rock"),
        (3, "Queen", "Bohemian Rhapsody", "Rock"),
        (4, "David Bowie", "Space Oddity", "Rock"),
        (5, "Queen", "Another One Bites the Dust", "Rock"),
        (6, "The Who", "My Generation", "Rock"),
        (7, "Led Zeppelin", "Stairway to Heaven", "Rock"),
        (8, "Jimi Hendrix", "Purple Haze", "Rock"),
        (9, "The Rolling Stones", "Paint It Black", "Rock"),
        (10, "Black Sabbath", "Paranoid", "Heavy Metal"),
        (11, "Iron Maiden", "Number of the Beast", "Heavy Metal"),
        (12, "Metallica", "Enter Sandman", "Heavy Metal"),
        (13, "Slayer", "Raining Blood", "Heavy Metal"),
        (14, "Judas Priest", "Breaking the Law", "Heavy Metal"),
        (15, "Ozzy Osbourne", "Crazy Train", "Heavy Metal"),
        (16, "Radiohead", "Creep", "Alternative"),
        (17, "Nirvana", "Smells Like Teen Spirit", "Grunge"),
        (18, "Pearl Jam", "Black", "Grunge"),
        (19, "Soundgarden", "Black Hole Sun", "Grunge"),
        (20, "Alice in Chains", "Man in the Box", "Grunge"),
    ]
    
    for id_val, artista, cancion, genero in canciones:
        tm.sql(f"INSERT INTO canciones VALUES ({id_val}, '{artista}', '{cancion}', '{genero}');")
    
    print(f"   ✓ Insertadas {len(canciones)} canciones\n")
   
    print("3. Construyendo índice invertido en 'cancion'...")
    tm.sql("BUILD TEX_INV ON canciones(cancion);")
    print("   ✓ Índice TEX_INV construido\n")
    
    print("4. Buscando canciones similares a 'Hey")
    results = tm.sql("""
    SELECT artista, cancion, genero FROM canciones    
    WHERE cancion @@ 'Hey'   
    LIMIT 9;
    """)
    

    
    for i, row in enumerate(results[0]['data'], 1):
        print(f"{i}. {row['artista']} - {row['cancion']} ({row['genero']})")
    
    print(f"\n{'='*60}")
    print("✓ Test completado exitosamente")
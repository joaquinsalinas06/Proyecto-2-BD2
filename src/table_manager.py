import os
import csv
import json
import inspect
import glob
from typing import List, Dict, Any, Optional
from .parser.ast import (
    ColumnDef, IndexType, Value, Condition, DataType,
    CompCond, BetweenCond, SpatialInCond,
    SpatialKNNCond, MultimediaKNNCond, LogicCond, Statement,
    CreateTableStmt, CreateTableFileStmt, SelectStmt,
    InsertStmt, DeleteStmt, IndexSpec
)
from .parser.sql_parser import SQLParser
from src.records import DynamicRecord
from src.records.indices import create_index
from src.multimedia.feature_extractors import SIFTExtractor
from src.multimedia.feature_extractors import MFCCExtractor
from src.multimedia.build_pipeline import build_knn_index_from_collection

'''
La clase Table representa una tabla en la base de datos, con su esquema y sus índices, de tal forma
que siempre tenemos conocimiento a los datos de dicha tabla, como nombre, columnas, tipos de datos, etc.

create_record_from_values: Dada una lista de valores, crea un DynamicRecord con los valores asignados a las columnas
get_primary_index: Retorna el índice primario de la tabla, si no existe lanza un error
'''
class Table:
    def __init__(self, name: str, columns: List[ColumnDef]):
        self.name = name
        self.columns = columns
        self.column_names = [col.name for col in columns]
        self.schema = columns 

        self.indexes = {}

        self.key_column = None
        for col in columns:
            if col.is_key:
                self.key_column = col.name
                break

    def create_record_from_values(self, values: List[Value]):
        if len(values) != len(self.columns):
            raise ValueError(f"Se esperaban {len(self.columns)} valores, se recibieron {len(values)}")
        
        kwargs = {}
        for col, value in zip(self.columns, values):
            val = value.value
            if hasattr(val, 'x') and hasattr(val, 'y'):
                val = (val.x, val.y)
            kwargs[col.name] = val

        return DynamicRecord(self.schema, **kwargs)
    
    def get_primary_index(self):
        if not self.key_column or self.key_column not in self.indexes:
            raise ValueError(f"No hay índice primario para la tabla '{self.name}'")
        return self.indexes[self.key_column]


class TableManager:
    def __init__(self):
        self.tables: Dict[str, Table] = {}
        self.indices_directory = self._detect_indices_directory()
        os.makedirs(self.indices_directory, exist_ok=True)
        self.metadata_file = os.path.join(self.indices_directory, "tables_metadata.json")

        self.parser = SQLParser()
        self._load_table_metadata()

    def _detect_indices_directory(self) -> str:
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        
        for frame_info in inspect.stack()[1:]:
            caller_file = frame_info.filename
            caller_file_normalized = caller_file.replace('\\', '/')

            if '/tests/' in caller_file_normalized:
                return os.path.join(project_root, "data", "test", "indices")

            if '/benchmarks/' in caller_file_normalized:
                return os.path.join(project_root, "data", "benchmarks", "indices")

        return os.path.join(project_root, "indices")

    def _get_multimedia_config(self, col: ColumnDef, table_name: str, csv_basename: str = None):
        if col.data_type not in [DataType.IMAGE, DataType.AUDIO]:
            return None, None, None

        options = col.index_options or {}

        if col.data_type == DataType.IMAGE:
            extractor = SIFTExtractor()
            default_clusters = 2000
        elif col.data_type == DataType.AUDIO:
            extractor = MFCCExtractor()
            default_clusters = 500
        else:
            return None, None, None

        vocabulary_size = options.get('clusters', default_clusters)

        # CASO 1: Dataset pre-construido
        if 'dataset' in options:
            dataset_name = options['dataset']
            if col.data_type == DataType.IMAGE and dataset_name == 'fashion':
                codebook_file = os.path.join(self.indices_directory, f"fashion_k{vocabulary_size}_codebook.dat")
            elif col.data_type == DataType.AUDIO and dataset_name == 'fma':
                codebook_file = os.path.join(self.indices_directory, f"fma_k{vocabulary_size}_codebook.dat")
            else:
                raise ValueError(f"Dataset '{dataset_name}' no reconocido para tipo {col.data_type.value}")

            if not os.path.exists(codebook_file):
                raise FileNotFoundError(
                    f"Codebook no encontrado: {codebook_file}\n"
                    f"Ejecuta 'python build_image_pipeline.py' para generar el codebook."
                )

            return vocabulary_size, codebook_file, extractor

        elif 'index_prefix' in options:
            index_prefix = options['index_prefix']

            if '_k' in index_prefix:
                try:
                    k_part = index_prefix.split('_k')[1]
                    vocabulary_size = int(k_part)
                except (IndexError, ValueError):
                    pass


            # Determina que codebook buscar
            codebook_candidates = [
                os.path.join(self.indices_directory, f"codebook_SIFT_k{vocabulary_size}.dat"),
                os.path.join(self.indices_directory, f"codebook_MFCC_k{vocabulary_size}.dat"),
            ]

            codebook_file = None
            for candidate in codebook_candidates:
                if os.path.exists(candidate):
                    codebook_file = candidate
                    break

            if not codebook_file:
                raise FileNotFoundError(
                    f"No se encontró codebook para index_prefix='{index_prefix}'\n"
                    f"Se buscó en: {codebook_candidates}\n"
                    f"Asegúrate de ejecutar 'python build_all_indices.py' primero."
                )

            return vocabulary_size, codebook_file, extractor

        # CASO 3: Construir desde collection
        elif 'collection' in options:
            collection_path = options['collection']
            if not os.path.exists(collection_path):
                raise FileNotFoundError(f"Collection path no encontrado: {collection_path}")

            # Incluir nombre del CSV si existe
            if csv_basename:
                base_filename = f"{table_name}_{col.name}_{csv_basename}"
            else:
                base_filename = f"{table_name}_{col.name}"

            codebook_file = os.path.join(
                self.indices_directory,
                f"{base_filename}_k{vocabulary_size}_codebook.dat"
            )
            return vocabulary_size, codebook_file, extractor

        else:
            raise ValueError(
                f"Índice KNN requiere 'dataset', 'index_prefix', o 'collection' en index_options.\n"
                f"Ejemplos:\n"
                f"  INDEX KNN_INV(dataset='fashion')\n"
                f"  INDEX KNN_INV(index_prefix='fashion_basic_k2000')\n"
                f"  INDEX KNN_SEQ(collection='data/imagenes', clusters=1000)"
            )

    def _build_codebook_from_collection(self, collection_path: str, codebook_file: str,
                                       vocabulary_size: int, extractor, inverted_file: str,
                                       mapping_file: str):
        histograms_file = inverted_file.replace('_inverted.dat', '_histograms.dat')
        buckets_file = inverted_file.replace('_inverted.dat', '_buckets.dat')

        build_knn_index_from_collection(
            collection_path=collection_path,
            codebook_file=codebook_file,
            vocabulary_size=vocabulary_size,
            extractor=extractor,
            inverted_file=inverted_file,
            mapping_file=mapping_file,
            histograms_file=histograms_file,
            buckets_file=buckets_file,
            verbose=True
        )


    #Se manda directamente el query a la funcion sql, que se encarga de parsearlo y ejecutar cada stmt
    def sql(self, query: str) -> List[Dict[str, Any]]:
        statements = self.parser.parse(query)
        results = []

        for stmt in statements:
            result = self._execute_statement(stmt)
            results.append(result)

        return results

    # Para cada uno de los stmts, se verifica que tipo es y se llama a la funcion/operacion correcta
    def _execute_statement(self, stmt: Statement) -> Dict[str, Any]: 
        try:
            if isinstance(stmt, CreateTableStmt):
                self.create_table(stmt.table_name, stmt.columns)
                return {
                    "type": "create_table",
                    "message": f"Tabla '{stmt.table_name}' creada exitosamente",
                    "table_name": stmt.table_name
                }

            elif isinstance(stmt, CreateTableFileStmt):
                self.create_table_from_file(stmt.table_name, stmt.file_path, stmt.indexes)
                return {
                    "type": "create_table_from_file",
                    "message": f"Tabla '{stmt.table_name}' creada desde archivo",
                    "table_name": stmt.table_name
                }

            elif isinstance(stmt, InsertStmt):
                if self.insert(stmt.table_name, stmt.values):
                    print(f"Registro insertado en '{stmt.table_name}'")
                return {
                    "type": "insert",
                    "message": f"Registro insertado en '{stmt.table_name}'",
                    "table_name": stmt.table_name
                }

            elif isinstance(stmt, SelectStmt):
                data = self.select(
                    stmt.table_name,
                    stmt.columns,
                    stmt.where_condition,
                    stmt.order_by,
                    stmt.order_desc,
                    stmt.limit
                )
                return {
                    "type": "select",
                    "data": data,
                    "table_name": stmt.table_name,
                    "rows_count": len(data)
                }

            elif isinstance(stmt, DeleteStmt):
                deleted_count = self.delete(stmt.table_name, stmt.where_condition)
                return {
                    "type": "delete",
                    "message": f"{deleted_count} registros eliminados de '{stmt.table_name}'",
                    "table_name": stmt.table_name,
                    "deleted_count": deleted_count
                }

            else:
                return {
                    "error": f"Tipo de sentencia no soportado:",
                    "type": "execution_error"
                }

        except NotImplementedError:
            raise

        except Exception as e:
            return {
                "error": str(e),
                "type": "Error ejecutando sentencia",
                "statement_type": type(stmt).__name__
            }
        
    '''
    Por cada una de las columnas que tenemos, verificamos si tiene un indice asociado
    aqui diferenciamos si es un indice primario o secundario
    en caso identificamos una columna que es llave primaria pero no tiene indice, le asignamos un BTree por defecto
    '''
    def create_table(self, table_name: str, columns: List[ColumnDef], csv_basename: str = None):
        if table_name in self.tables:
            raise ValueError(f"La tabla '{table_name}' ya existe")
        
        table = Table(table_name, columns)
        self.tables[table_name] = table

        for col in table.columns:
            if col.index_type:
                if col.data_type in [DataType.IMAGE, DataType.AUDIO]:
                    vocabulary_size, codebook_file, extractor = self._get_multimedia_config(col, table.name)
                    col.vocabulary_size = vocabulary_size
                    
                    options = col.index_options or {}
                    if 'collection' in options and not os.path.exists(codebook_file):
                        collection_path = options['collection']
                        # Incluir nombre del CSV si existe
                        if csv_basename:
                            base_filename = f"{table.name}_{col.name}_{csv_basename}"
                        else:
                            base_filename = f"{table.name}_{col.name}"
                        inverted_file = os.path.join(self.indices_directory, f"{base_filename}_inverted.dat")
                        mapping_file = os.path.join(self.indices_directory, f"{base_filename}_mapping.dat")
                        histograms_file = inverted_file.replace('_inverted.dat', '_histograms.dat')
                        buckets_file = inverted_file.replace('_inverted.dat', '_buckets.dat')

                        id_extractor = None
                        if 'id_from_filename' in options and options['id_from_filename']:
                            id_extractor = lambda path: int(os.path.basename(path).split('.')[0])

                        build_knn_index_from_collection(
                            collection_path=collection_path,
                            codebook_file=codebook_file,
                            vocabulary_size=vocabulary_size,
                            extractor=extractor,
                            inverted_file=inverted_file,
                            mapping_file=mapping_file,
                            histograms_file=histograms_file,
                            buckets_file=buckets_file,
                            verbose=True,
                            id_extractor=id_extractor
                        )
                else:
                    vocabulary_size, codebook_file, extractor = None, None, None
                
                # Extraer index_prefix si existe en options
                index_prefix = None
                if col.index_options and 'index_prefix' in col.index_options:
                    index_prefix = col.index_options['index_prefix']

                index = create_index(
                    col.index_type,
                    col.name,
                    filename=os.path.join(self.indices_directory, f"{table.name}_{col.name}.dat"),
                    is_primary=col.is_key,
                    primary_key_column=table.key_column if not col.is_key else None,
                    table_schema=table.columns,
                    vocabulary_size=vocabulary_size,
                    codebook_file=codebook_file,
                    extractor=extractor,
                    index_prefix=index_prefix
                )
                table.indexes[col.name] = index
            elif col.is_key and col.index_type is None:
                col.index_type = IndexType.BTREE
                index = create_index(
                    col.index_type,
                    col.name,
                    filename=os.path.join(self.indices_directory, f"{table.name}_{col.name}.dat"),
                    is_primary=True,
                    primary_key_column=None,
                    table_schema=table.columns
                )
                table.indexes[col.name] = index

        self._save_table_metadata()
    '''
    A partir de un archivo CSV, realizamos dos pasadas en streaming (sin cargar todo en memoria):
    1. Primera pasada: analizar tipos de datos y determinar tamaño óptimo para VARCHAR
    2. Segunda pasada: insertar registros en los índices
    '''
    def create_table_from_file(self, table_name: str, file_path: str, indexes: List['IndexSpec']):
        if table_name in self.tables:
            raise ValueError(f"La tabla '{table_name}' ya existe")

        csv_basename = os.path.splitext(os.path.basename(file_path))[0]

        # Extraer la columna primaria y crear diccionario de índices por columna
        primary_index_spec = None
        index_specs_by_column = {}

        for index_spec in indexes:
            index_specs_by_column[index_spec.column_name] = index_spec
            if index_spec.is_primary:
                primary_index_spec = index_spec

        if not primary_index_spec:
            raise ValueError("Debe especificar un índice primario")

        column_stats = {}

        # Primera pasada: Obtenemos estadísticas de las columnas
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            reader = csv.DictReader(file)
            headers = reader.fieldnames

            if not headers:
                raise ValueError(f"Archivo '{file_path}' vacío")

            for header in headers:
                column_stats[header] = {
                    'type': DataType.INT,
                    'max_length': 0
                }

            row_count = 0
            for row in reader:
                row_count += 1
                for header in headers:
                    value = row[header].strip()
                    stats = column_stats[header]

                    # Detectar formato de ubicación "(lat,lon)"
                    if value.startswith('(') and value.endswith(')') and ',' in value:
                        stats['type'] = DataType.ARRAY
                        continue

                    if value: 
                        is_int = value.isdigit() or (value.startswith('-') and len(value) > 1 and value[1:].isdigit())
                        is_float = value.lstrip('-').replace('.', '', 1).isdigit() and '.' in value and value.lstrip('-').replace('.', '', 1)

                        if is_int and stats['type'] == DataType.INT:
                            pass
                        elif is_float:
                            if stats['type'] == DataType.INT:
                                stats['type'] = DataType.FLOAT
                        else:
                            stats['type'] = DataType.VARCHAR
                            stats['max_length'] = max(stats['max_length'], len(value))
                    if stats['type'] == DataType.VARCHAR and value:
                        stats['max_length'] = max(stats['max_length'], len(value))

            if row_count == 0:
                raise ValueError(f"Archivo '{file_path}' sin datos")

        # Construimos el esquema con índices
        columns = []
        for header in headers:
            is_key = (header == primary_index_spec.column_name)
            index_spec = index_specs_by_column.get(header)
            col_index_type = index_spec.index_type if index_spec else None
            stats = column_stats[header]

            size = None
            element_type = None
            array_dimensions = None
            data_type = stats['type']

            # Check if index options specify a type override (for multimedia columns)
            type_overridden = False
            if index_spec and index_spec.index_options:
                type_override = index_spec.index_options.get('type')
                if type_override:
                    if type_override.upper() == 'IMAGE':
                        data_type = DataType.IMAGE
                        size = 200  # Default size for IMAGE paths
                        type_overridden = True
                    elif type_override.upper() == 'AUDIO':
                        data_type = DataType.AUDIO
                        size = 200  # Default size for AUDIO paths
                        type_overridden = True

            # Only apply default sizing if type wasn't overridden
            if not type_overridden:
                if stats['type'] == DataType.VARCHAR:
                    size = max(1, int(stats['max_length'] * 1.1))
                elif stats['type'] == DataType.ARRAY:
                    element_type = DataType.FLOAT
                    array_dimensions = 2

            if index_spec:
                col_index_options = index_spec.index_options  
            else:   
                col_index_options = None

            column = ColumnDef(
                name=header,
                data_type=data_type,
                size=size,
                element_type=element_type,
                is_key=is_key,
                index_type=col_index_type,
                array_dimensions=array_dimensions,
                index_options=col_index_options
            )
            columns.append(column)

        table = Table(table_name, columns)
        self.tables[table_name] = table

        for col in table.columns:
            if col.index_type:
                is_knn_index = col.index_type in [IndexType.KNN_SEQ, IndexType.KNN_INV]
                is_multimedia_type = col.data_type in [DataType.IMAGE, DataType.AUDIO]

                if is_knn_index or is_multimedia_type:
                    # Para KNN desde CSV, incluir nombre del CSV en archivos
                    options = col.index_options or {}

                    if not is_multimedia_type and 'collection' in options:
                        collection_path = options['collection']
                        if any(ext in collection_path.lower() for ext in ['jpg', 'jpeg', 'png', 'image']):
                            extractor = SIFTExtractor()
                            default_clusters = 2000
                        else:
                            extractor = MFCCExtractor()
                            default_clusters = 500

                        vocabulary_size = options.get('k', default_clusters)
                        base_filename = f"{table.name}_{col.name}_{csv_basename}"
                        codebook_file = os.path.join(
                            self.indices_directory,
                            f"{base_filename}_k{vocabulary_size}_codebook.dat"
                        )
                    else:
                        vocabulary_size, codebook_file, extractor = self._get_multimedia_config(col, table.name, csv_basename)

                    col.vocabulary_size = vocabulary_size

                    if 'collection' in options and not os.path.exists(codebook_file):
                        collection_path = options['collection']
                        base_filename = f"{table.name}_{col.name}_{csv_basename}"
                        inverted_file = os.path.join(self.indices_directory, f"{base_filename}_inverted.dat")
                        mapping_file = os.path.join(self.indices_directory, f"{base_filename}_mapping.dat")
                        histograms_file = inverted_file.replace('_inverted.dat', '_histograms.dat')
                        buckets_file = inverted_file.replace('_inverted.dat', '_buckets.dat')

                        id_extractor = None
                        if options.get('id_from_filename'):
                            id_extractor = lambda path: int(os.path.basename(path).split('.')[0])

                        build_knn_index_from_collection(
                            collection_path=collection_path,
                            codebook_file=codebook_file,
                            vocabulary_size=vocabulary_size,
                            extractor=extractor,
                            inverted_file=inverted_file,
                            mapping_file=mapping_file,
                            histograms_file=histograms_file,
                            buckets_file=buckets_file,
                            verbose=True,
                            id_extractor=id_extractor
                        )
                else:
                    vocabulary_size, codebook_file, extractor = None, None, None

                # Extraer index_prefix si existe en options
                index_prefix = None
                if col.index_options and 'index_prefix' in col.index_options:
                    index_prefix = col.index_options['index_prefix']

                index = create_index(
                    col.index_type,
                    col.name,
                    filename=os.path.join(self.indices_directory, f"{table.name}_{col.name}.dat"),
                    is_primary=col.is_key,
                    primary_key_column=table.key_column if not col.is_key else None,
                    table_schema=table.columns,
                    expected_size=row_count,
                    vocabulary_size=vocabulary_size,
                    codebook_file=codebook_file,
                    extractor=extractor,
                    index_prefix=index_prefix
                )
                table.indexes[col.name] = index

        # Segunda pasada: insertar registros en índices
        all_records = []
        skipped_records = 0
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            reader = csv.DictReader(file)
            current_row = 0
            for row in reader:
                current_row += 1
                parsed_row = {}
                skip_record = False
                
                for col in table.columns:
                    value_str = row[col.name].strip() if row[col.name] else ""

                    if col.data_type == DataType.ARRAY:
                        if value_str.startswith('(') and value_str.endswith(')'):
                            coords_str = value_str[1:-1]
                            coords = [float(x.strip()) for x in coords_str.split(',')]
                            parsed_row[col.name] = tuple(coords)
                        else:
                            parsed_row[col.name] = value_str
                    elif col.data_type == DataType.INT:
                        try:
                            int_value = int(value_str) if value_str else 0
                            if int_value > 2147483647 or int_value < -2147483648:
                                skip_record = True
                                break
                            parsed_row[col.name] = int_value
                        except ValueError:
                            parsed_row[col.name] = value_str
                    elif col.data_type == DataType.BIGINT:
                        try:
                            parsed_row[col.name] = int(value_str) if value_str else 0
                        except ValueError:
                            parsed_row[col.name] = value_str
                    elif col.data_type == DataType.FLOAT:
                        try:
                            parsed_row[col.name] = float(value_str) if value_str else 0.0
                        except ValueError:
                            parsed_row[col.name] = value_str
                    else:
                        parsed_row[col.name] = value_str

                if not skip_record:
                    all_records.append(parsed_row)
                else:
                    skipped_records += 1


        # Cuando cargamos muchos registros, es mejor hacer bulk load en los indices primarios
        for _, index in table.indexes.items():
            if index is not None:
                # Saltar índices que no soportan escritura (KNN_INV, KNN_SEQ)
                if not hasattr(index, 'add') and not hasattr(index, 'bulk_load'):
                    continue

                # Si es que existe el metodo de bulk load lo usaremos, como en Sequential, ISAM o B+Tree
                if hasattr(index, 'bulk_load'):
                    index.bulk_load(all_records)
                elif hasattr(index, 'add'):
                    # Para los indices secundarios usaremos solo el add regular, por sus propiedades
                    for _, record in enumerate(all_records, 1):
                        index.add(record)

        self._save_table_metadata()

    '''
    Insertamos un registro en la tabla, verificando que la tabla exista
    Por cada columna que tenga un indice, insertamos el registro en el indice
    '''
    def insert(self, table_name: str, values: List[Value]):
        if table_name not in self.tables:
            raise ValueError(f"La tabla '{table_name}' no existe")

        table = self.tables[table_name]

        for col_name, index in table.indexes.items():
            if index is not None and not hasattr(index, 'add'):
                raise NotImplementedError(
                    f"INSERT no soportado: la tabla '{table_name}' tiene índices multimedia que son solo lectura. "
                    f"Los índices multimedia trabajan con colecciones pre-construidas."
                )

        record_dict = {}
        for col, value in zip(table.columns, values):
            record_dict[col.name] = value.value


        for _, index in table.indexes.items():
            if index is not None and hasattr(index, 'add'):
                index.add(record_dict)
        return True
                


    '''
    Al realizar una busqueda, verificaremos si es que existe algun tipo de condicion para filtrar los registros
    luego si es que existe algun orden en especifico y finalmente si tenemos un limite de registros a retornar
    Si se seleccionan todas las columnas, retornamos todos los registros, sino solo las columnas especificadas
    '''
    def select(self, table_name: str, columns: List[str],
              where_condition: Optional[Condition] = None,
              order_by: Optional[str] = None,
              order_desc: bool = False,
              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        
        if table_name not in self.tables:
            raise ValueError(f"La tabla '{table_name}' no existe")
        table = self.tables[table_name]

        if where_condition:
            filtered_dicts = self._execute_condition(table, where_condition, limit=limit)
        else:
            index = table.get_primary_index()
            filtered_dicts = index.getAllRecords()

        if order_by:
            filtered_dicts.sort(key=lambda d: d.get(order_by), reverse=order_desc)

        if limit:
            filtered_dicts = filtered_dicts[:limit]

        select_all = "*" in columns # Verificar si se seleccionan todas las columnas
        results = []

        for record in filtered_dicts:
            formatted_record = {}

            for key, value in record.items():
                if not select_all and key not in columns: # Verificar si la columna está en la selección
                    continue

                if isinstance(value, (tuple, list)): # Formatear arrays como "(val1, val2)"
                    formatted_record[key] = f"({', '.join(str(v) for v in value)})"
                else:
                    formatted_record[key] = value

            results.append(formatted_record)

        return results

    '''
    Si no existe un filtro de borrado, se eliminan todos los registros de la tabla
    Si existe un filtro, se obtienen los registros a eliminar y por cada uno de ellos
    se eliminan de los indices asociados a la tabla, en cada uno d elos indices y se devuelve
    la cantidad de registros eliminados
    '''
    def delete(self, table_name: str, where_condition: Optional[Condition] = None) -> int:
        if table_name not in self.tables:
            raise ValueError(f"La tabla '{table_name}' no existe")
        table = self.tables[table_name]

        for col_name, index in table.indexes.items():
            if index is not None and not hasattr(index, 'remove'):
                raise NotImplementedError(
                    f"DELETE no soportado: la tabla '{table_name}' tiene índices multimedia que son solo lectura. "
                    f"Los índices multimedia trabajan con colecciones pre-construidas."
                )

        if where_condition is None:
            deleted_count = 0
            for col_name, index in table.indexes.items():
                if index is not None and hasattr(index, 'clear_all'):
                    deleted_count = index.clear_all()
            return deleted_count

        records_to_delete = self._execute_condition(table, where_condition)
        deleted_count = 0

        for record_dict in records_to_delete:
            deleted_count += 1
            for col_name, index in table.indexes.items():
                if index is not None and hasattr(index, 'remove'):
                    key = record_dict.get(col_name)
                    if not index.is_primary and table.key_column:
                        pk_value = record_dict.get(table.key_column)
                        index.remove(key, primary_key=pk_value)
                    else:
                        index.remove(key)
        return deleted_count

    '''
    Este es el core de la obtencion de registros con condiciones
    Dependiendo del tipo de condicion, se realiza la busqueda correspondiente
    '''
    def _execute_condition(self, table: Table, condition: Condition, limit: Optional[int] = None) -> List:
        if isinstance(condition, CompCond):
            return self._comparison_condition(table, condition)

        elif isinstance(condition, BetweenCond):
            return self._between_condition(table, condition)

        elif isinstance(condition, SpatialInCond):
            return self._spatial_in_condition(table, condition)

        elif isinstance(condition, SpatialKNNCond):
            return self._spatial_knn_condition(table, condition)

        elif isinstance(condition, MultimediaKNNCond):
            return self._multimedia_knn_condition(table, condition, limit)

        elif isinstance(condition, LogicCond):
            return self._logic_condition(table, condition)

        return table.get_primary_index().getAllRecords()

    '''
    Aqui se aprovechan principalmente los indices para realizar las busquedas
    Si no existe un indice para la columna, se realiza un escaneo completo de la tabla y se filtran los datos
    En el caso de los indices secundarios, se obtiene una referencia (la llave primaria) y se vuelve a realizar
    una busqueda en el indice primario para obtener el registro completo
    '''
    def _comparison_condition(self, table: Table, condition: CompCond) -> List:
        column_name = condition.column
        operator = condition.operator.value
        search_value = condition.value.value

        index = table.indexes.get(column_name)

        if index is None:
            all_records = table.get_primary_index().getAllRecords()
            matching_records = []
            for record in all_records:
                record_value = record.get(column_name)
                match = False

                if operator == "=" and record_value == search_value:
                    match = True
                elif operator == "!=" and record_value != search_value:
                    match = True
                elif operator == "<" and record_value < search_value:
                    match = True
                elif operator == "<=" and record_value <= search_value:
                    match = True
                elif operator == ">" and record_value > search_value:
                    match = True
                elif operator == ">=" and record_value >= search_value:
                    match = True

                if match:
                    matching_records.append(record)

            return matching_records

        try:
            if operator == "=":
                index_results = index.rangeSearch(search_value, search_value)
            elif operator == "<":
                index_results = index.rangeSearch(None, search_value, begin_inclusive=True, end_inclusive=False)
            elif operator == "<=":
                index_results = index.rangeSearch(None, search_value, begin_inclusive=True, end_inclusive=True)
            elif operator == ">":
                index_results = index.rangeSearch(search_value, None, begin_inclusive=False, end_inclusive=True)
            elif operator == ">=":
                index_results = index.rangeSearch(search_value, None, begin_inclusive=True, end_inclusive=True)
            elif operator == "!=":
                all_records = index.getAllRecords()
                filtered_records = []
                for record in all_records:
                    if record.get(index.column_name) != search_value:
                        filtered_records.append(record)
                return filtered_records
            else:
                raise ValueError(f"Operador no soportado {operator}")

            if index.is_primary:
                return index_results
            
            if not table.key_column:
                return []

            primary_key_values = []
            for reference in index_results:
                if table.key_column in reference:
                    pk_value = reference.get(table.key_column)
                    primary_key_values.append(pk_value)

            primary_index = table.indexes.get(table.key_column)
            if primary_index:
                full_records = []
                for pk_value in primary_key_values:
                    records = primary_index.search(pk_value)
                    full_records.extend(records)
                return full_records

            all_records = table.get_primary_index().getAllRecords()
            matching_records = []
            for record in all_records:
                if record.get(table.key_column) in primary_key_values:
                    matching_records.append(record)
            return matching_records

        except NotImplementedError:
            raise

        except Exception:
            all_records = table.get_primary_index().getAllRecords()
            matching_records = []
            for record in all_records:
                record_value = record.get(column_name)
                match = False

                if operator == "=" and record_value == search_value:
                    match = True
                elif operator == "!=" and record_value != search_value:
                    match = True
                elif operator == "<" and record_value < search_value:
                    match = True
                elif operator == "<=" and record_value <= search_value:
                    match = True
                elif operator == ">" and record_value > search_value:
                    match = True
                elif operator == ">=" and record_value >= search_value:
                    match = True

                if match:
                    matching_records.append(record)

            return matching_records

    '''
    Aqui aprovechamos principalmete el range search de los indices (Hash no soporta range search y RTree usa un tipo de rango espacial)
    Nuevamente si no existe indice se toman todos los registros y si es secundario se obtienen las referencias para buscar en el primario
    '''
    def _between_condition(self, table: Table, condition: BetweenCond) -> List:
        column_name = condition.column
        start_value = condition.start_value.value
        end_value = condition.end_value.value

        index = table.indexes.get(column_name)

        if index is None:
            all_records = table.get_primary_index().getAllRecords()
            matching_records = []
            for record in all_records:
                record_value = record.get(column_name)
                if start_value <= record_value <= end_value:
                    matching_records.append(record)
            return matching_records

        try:
            index_results = index.rangeSearch(start_value, end_value, begin_inclusive=True, end_inclusive=True)

            if index.is_primary:
                return index_results

            if not table.key_column:
                return []

            primary_key_values = []
            for reference in index_results:
                if table.key_column in reference:
                    pk_value = reference.get(table.key_column)
                    primary_key_values.append(pk_value)

            primary_index = table.indexes.get(table.key_column)
            if primary_index:
                full_records = []
                for pk_value in primary_key_values:
                    records = primary_index.search(pk_value)
                    full_records.extend(records)
                return full_records

            all_records = table.get_primary_index().getAllRecords()
            matching_records = []
            for record in all_records:
                if record.get(table.key_column) in primary_key_values:
                    matching_records.append(record)
            return matching_records

        except NotImplementedError:
            raise

        except Exception:

            all_records = table.get_primary_index().getAllRecords()
            matching_records = []
            for record in all_records:
                record_value = record.get(column_name)
                if start_value <= record_value <= end_value:
                    matching_records.append(record)
            return matching_records

    '''
    Esta busqueda es exclusiva de los indices espaciales (RTree), en este caso se le brinda un punto y un radio
    y mediante el metodo rangeSearch del indice, se obtienen las referencias a los registros que cumplen con la condicion
    Luego se busca en el indice primario para obtener los registros completos
    '''
    def _spatial_in_condition(self, table: Table, condition: SpatialInCond) -> List:
        column_name = condition.column
        point = condition.point
        radius = condition.radius

        index = table.indexes.get(column_name)
        if index is None:
            return []

        try:
            pk_references = index.rangeSearch(point, radius)
            primary_index = table.get_primary_index()

            full_records = []
            for pk_ref in pk_references:
                pk_value = pk_ref[table.key_column]
                records = primary_index.search(pk_value)
                full_records.extend(records)

            return full_records

        except Exception:
            return []

    '''
    Esta busqueda nuevamente es solo para los RTree, donde se le brinda un punto y un k, que es la cantidad de registros
    más cercanos a ese punto que se desean obtener
    Nuevamente se obtienen las referencias y se busca en el indice primario para obtener los registros completos
    '''
    def _spatial_knn_condition(self, table: Table, condition: SpatialKNNCond) -> List:
        column_name = condition.column
        point = condition.point
        k = condition.k

        index = table.indexes.get(column_name)
        if index is None:
            return []

        try:
            pk_references = index.knnSearch(point, k)
            primary_index = table.get_primary_index()

            full_records = []
            for pk_ref in pk_references:
                pk_value = pk_ref[table.key_column]
                records = primary_index.search(pk_value)
                full_records.extend(records)

            return full_records

        except Exception:
            return []

    def _multimedia_knn_condition(self, table: Table, condition: MultimediaKNNCond, limit: Optional[int] = None) -> List:
        column_name = condition.column
        query_path = condition.query_path
        k = limit if limit is not None else 10

        index = table.indexes.get(column_name)
        if index is None:
            raise ValueError(f"No existe índice KNN para la columna '{column_name}'")

        if not hasattr(index, 'knnSearchByFile'):
            raise ValueError(f"El índice de la columna '{column_name}' no soporta búsqueda KNN multimedia")

        try:
            results = index.knnSearchByFile(query_path, k)
            records_with_distance = []
            
            num_expected_columns = len(table.columns)
            primary_index = table.indexes.get(table.key_column)
            
            for record, distance in results:
                # KNN siempre retorna 2 keys: 'id' + column_name 
                num_record_columns = len([k for k in record.keys() if k != '_distance'])
                is_partial = num_record_columns < num_expected_columns
                
                #Si es que el numero de columnas es menor al esperado, se asume que es un registro parcial
                if is_partial and primary_index and table.key_column in record:
                    # Buscamos en el primario para obtener el registro completo
                    pk_value = record.get(table.key_column)
                    # Usar rangeSearch en lugar de search (bug en BTree search)
                    full_records = primary_index.rangeSearch(pk_value, pk_value)
                    
                    if full_records:
                        full_record = full_records[0].copy()
                        full_record['_distance'] = distance
                        records_with_distance.append(full_record)
                    else:
                        # Si no se encuentra en primario usamos lo obtenido
                        record_copy = record.copy()
                        record_copy['_distance'] = distance
                        records_with_distance.append(record_copy)
                else:
                    # El registro ya es completo
                    record_copy = record.copy()
                    record_copy['_distance'] = distance
                    records_with_distance.append(record_copy)
                    
            return records_with_distance
        except Exception as e:
            raise ValueError(f"Error en búsqueda KNN multimedia: {str(e)}")

    '''
    Esta funcion se encarga de manejar las condiciones logicas AND y OR, de tal forma que podamos encadenar multiples
    condiciones en una sola consulta
    '''
    def _logic_condition(self, table: Table, condition: LogicCond) -> List:
        left_results = self._execute_condition(table, condition.left)
        right_results = self._execute_condition(table, condition.right)

        if condition.operator.value == "AND":
            intersection_results = []
            for record in left_results:
                if record in right_results:
                    intersection_results.append(record)
            return intersection_results

        elif condition.operator.value == "OR":
            union_results = left_results.copy()
            for record in right_results:
                if record not in union_results:
                    union_results.append(record)
            return union_results

        return []

    def show_tables(self) -> List[str]:
        return list(self.tables.keys())

    '''
    Retorna metadatos de todas las tablas existentes: nombre, columnas, indices y cantidad de registros
    '''
    def get_all_tables_metadata(self) -> List[Dict[str, Any]]:
        tables_metadata = []

        for table_name, table in self.tables.items():
            columns_info = []
            indexed_columns = []

            # Extraemos informacion de cada columna
            for col in table.columns:
                col_info = {
                    "name": col.name,
                    "type": self._format_column_type(col),
                    "indexed": col.name in table.indexes,
                    "indexType": col.index_type.value if col.index_type else None,
                    "primaryKey": col.is_key
                }
                columns_info.append(col_info)

                if col.name in table.indexes:
                    indexed_columns.append(col.name)

            # Obtener cantidad de registros del indice primario
            row_count = 0
            if table.key_column and table.key_column in table.indexes:
                try:
                    primary_index = table.indexes[table.key_column]
                    row_count = len(primary_index.getAllRecords())
                except Exception as e:
                    row_count = 0

            tables_metadata.append({
                "name": table_name,
                "columns": columns_info,
                "row_count": row_count,
                "indexes": indexed_columns
            })

        return tables_metadata

    '''
    Retorna informacion detallada de una tabla especifica: esquema completo y muestra de los primeros registros
    '''
    def get_table_details(self, table_name: str, preview_limit: int = 100) -> Dict[str, Any]:
        if table_name not in self.tables:
            raise ValueError(f"La tabla '{table_name}' no existe")

        table = self.tables[table_name]

        # Construir informacion de cada columna
        columns_info = []
        for col in table.columns:
            columns_info.append({
                "name": col.name,
                "type": self._format_column_type(col),
                "indexed": col.name in table.indexes,
                "indexType": col.index_type.value if col.index_type else None,
                "primaryKey": col.is_key
            })

        # Obtener muestra de datos limitada
        preview_data = []
        if table.key_column and table.key_column in table.indexes:
            primary_index = table.indexes[table.key_column]
            all_records = primary_index.getAllRecords()
            preview_data = all_records[:preview_limit]

        return {
            "name": table_name,
            "description": f"Tabla '{table_name}' con {len(table.columns)} columnas",
            "columns": columns_info,
            "data": preview_data
        }

    '''
    Formatea el tipo de dato de una columna como string para mostrar al usuario
    Incluye el tamaño para VARCHAR y dimensiones para ARRAY
    '''
    def _format_column_type(self, col: ColumnDef) -> str:
        if col.data_type == DataType.VARCHAR and col.size:
            return f"VARCHAR[{col.size}]"
        elif col.data_type == DataType.ARRAY and col.array_dimensions:
            base_type = "FLOAT"
            return f"ARRAY[{col.array_dimensions}][{base_type}]"
        else:
            return col.data_type.value

    '''
    Guarda los metadatos de todas las tablas en un archivo JSON
    Permite reconstruir las tablas e indices al reiniciar el sistema
    Se usa JSON para facilidad de lectura y escritura con el frontend
    Y al contener solo metadatos, no es un archivo muy pesado
    '''
    def _save_table_metadata(self):
        metadata = {}

        # Por cada una de las tablas, guardamos su esquema y detalles de columnas
        for table_name, table in self.tables.items():
            columns_data = []
            for col in table.columns:
                col_data = {
                    "name": col.name,
                    "data_type": col.data_type.value,
                    "size": col.size,
                    "element_type": col.element_type.value if col.element_type else None,
                    "is_key": col.is_key,
                    "index_type": col.index_type.value if col.index_type else None,
                    "array_dimensions": col.array_dimensions,
                    "index_options": col.index_options
                }
                columns_data.append(col_data)

            metadata[table_name] = {
                "name": table_name,
                "columns": columns_data
            }

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    '''
    Carga los metadatos desde el archivo JSON y reconstruye todas las tablas con sus indices
    Se ejecuta al iniciar el TableManager para recuperar el estado previo del sistema
    '''
    def _load_table_metadata(self):
        if not os.path.exists(self.metadata_file):
            return

        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)

        for table_name, table_data in metadata.items():
            # Reconstruir columnas desde metadatos
            columns = []
            for col_data in table_data["columns"]:
                col = ColumnDef(
                    name=col_data["name"],
                    data_type=DataType(col_data["data_type"]),
                    size=col_data.get("size"),
                    element_type=DataType(col_data["element_type"]) if col_data.get("element_type") else None,
                    is_key=col_data.get("is_key", False),
                    index_type=IndexType(col_data["index_type"]) if col_data.get("index_type") else None,
                    array_dimensions=col_data.get("array_dimensions"),
                    index_options=col_data.get("index_options")
                )
                columns.append(col)

            # Crear tabla y reconstruir sus indices desde archivos existentes
            table = Table(table_name, columns)
            primary_key_col = table.key_column

            for col in columns:
                if col.index_type:
                    filename = os.path.join(self.indices_directory, f"{table_name}_{col.name}.dat")

                    # Verificar si es índice KNN (por tipo de dato o por tipo de índice)
                    is_knn_index = col.index_type in [IndexType.KNN_SEQ, IndexType.KNN_INV]
                    is_multimedia_type = col.data_type in [DataType.IMAGE, DataType.AUDIO]

                    if is_knn_index or is_multimedia_type:
                        options = col.index_options or {}

                        # Inferir extractor desde collection path
                        if 'collection' in options:
                            collection_path = options['collection']
                            if any(ext in collection_path.lower() for ext in ['jpg', 'jpeg', 'png', 'image']):
                                extractor = SIFTExtractor()
                            else:
                                extractor = MFCCExtractor() if MFCCExtractor else None

                            vocabulary_size = options.get('k', 2000)

                            pattern = os.path.join(
                                self.indices_directory,
                                f"{table_name}_{col.name}_*_k{vocabulary_size}_codebook.dat"
                            )
                            matches = glob.glob(pattern)
                            if matches:
                                codebook_file = matches[0]
                            else:
                                codebook_file = None
                        else:
                            vocabulary_size, codebook_file, extractor = self._get_multimedia_config(col, table_name)
                    else:
                        vocabulary_size, codebook_file, extractor = None, None, None

                    # Extraer index_prefix si existe en options
                    index_prefix = None
                    if col.index_options and 'index_prefix' in col.index_options:
                        index_prefix = col.index_options['index_prefix']

                    index = create_index(
                        index_type=col.index_type,
                        column_name=col.name,
                        filename=filename,
                        is_primary=col.is_key,
                        primary_key_column=primary_key_col if not col.is_key else None,
                        table_schema=table.columns,
                        vocabulary_size=vocabulary_size,
                        codebook_file=codebook_file,
                        extractor=extractor,
                        index_prefix=index_prefix
                    )
                    table.indexes[col.name] = index

            self.tables[table_name] = table
    
    def get_io_stats(self, table_name: str) -> Dict[str, int]:
        if table_name not in self.tables:
            return {}
        
        table = self.tables[table_name]
        primary_index = table.indexes.get(table.key_column)
        
        if primary_index and hasattr(primary_index, 'get_io_stats'):
            return primary_index.get_io_stats()
        
        return {}
    
    def reset_io_stats(self, table_name: str):
        if table_name not in self.tables:
            return
        
        table = self.tables[table_name]
        primary_index = table.indexes.get(table.key_column)
        
        if primary_index and hasattr(primary_index, 'reset_io_stats'):
            primary_index.reset_io_stats()
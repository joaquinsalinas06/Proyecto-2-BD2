from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from datetime import datetime
from typing import List
import time
import os
import uuid
import json
import mimetypes
from src.table_manager import TableManager
from api.models import (
    QueryRequest, QueryResponse, TableListResponse,
    TableDetailsResponse, QueryHistoryResponse, QueryHistoryItem,
    UploadedFile, UploadedFilesResponse, UploadResponse
)

router = APIRouter()

# Lazy loading del TableManager para evitar problemas con volúmenes montados
_tm = None

def get_tm():
    global _tm
    if _tm is None:
        _tm = TableManager()
    return _tm

query_history: List[QueryHistoryItem] = []

# Configuración para archivos subidos
UPLOADS_DIR = "data/api/uploads"
UPLOADS_REGISTRY = "data/api/uploads_registry.json"
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}

# Asegurar que el directorio de uploads existe
os.makedirs(UPLOADS_DIR, exist_ok=True)

def _load_uploads_registry() -> List[dict]:
    """Carga el registro de archivos subidos"""
    if os.path.exists(UPLOADS_REGISTRY):
        with open(UPLOADS_REGISTRY, 'r') as f:
            return json.load(f)
    return []

def _save_uploads_registry(files: List[dict]):
    """Guarda el registro de archivos subidos"""
    os.makedirs(os.path.dirname(UPLOADS_REGISTRY), exist_ok=True)
    with open(UPLOADS_REGISTRY, 'w') as f:
        json.dump(files, f, indent=2)

def _get_file_type(filename: str) -> str:
    """Determina si el archivo es IMAGE o AUDIO"""
    ext = os.path.splitext(filename)[1].lower()
    if ext in ALLOWED_IMAGE_EXTENSIONS:
        return "IMAGE"
    elif ext in ALLOWED_AUDIO_EXTENSIONS:
        return "AUDIO"
    return "UNKNOWN"

# Se obtienen todas las tablas disponibles
@router.get("/tables", response_model=TableListResponse)
async def get_tables():
    try:
        tables_metadata = get_tm().get_all_tables_metadata()
        return TableListResponse(tables=tables_metadata)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Error en /tables endpoint:")
        print(error_details)
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener tablas: {str(e)}"
        )

# Se obtienen los detalles de una tabla específica
@router.get("/tables/{table_name}", response_model=TableDetailsResponse)
async def get_table_details(table_name: str, limit: int = 25):
    try:
        table_details = get_tm().get_table_details(table_name, preview_limit=limit)
        return TableDetailsResponse(**table_details)
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener detalles de tabla: {str(e)}"
        )

#Mostrar el historial de consultas
@router.get("/history", response_model=QueryHistoryResponse)
async def get_query_history():
    return QueryHistoryResponse(history=query_history)

@router.post("/execute", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    query_preview = request.query[:100] + "..." if len(request.query) > 100 else request.query

    # LOG: Ver qué query llega
    print(f"[DEBUG] Query recibida: '{request.query}'")
    print(f"[DEBUG] Query stripped: '{request.query.strip()}'")

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query no puede estar vacía"
        )

    #Iniciamos una medicion de tiempo para ver el tiempo de ejecucion
    start_time = time.time()
    try:
        print("owo")
        results = get_tm().sql(request.query)
        print("uwu")
        print(results)
    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000

        # LOG: Ver el error exacto
        print(f"[ERROR] Query falló: {str(e)}")
        print(f"[ERROR] Tipo: {type(e).__name__}")

        _add_to_history(request.query, execution_time_ms, success=False)

        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": str(e),
                "query_preview": query_preview
            }
        )
    execution_time_ms = (time.time() - start_time) * 1000

    # LOG: Ver resultados
    print(f"[DEBUG] Resultados: {results}")

    has_errors = any(result.get("error") for result in results)

    if has_errors:
        error_result = next(result for result in results if result.get("error"))
        print(f"[ERROR] Error en resultado: {error_result['error']}")
        print(f"[ERROR] Tipo: {error_result.get('type', 'unknown')}")
        
        _add_to_history(request.query, execution_time_ms, success=False)
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": error_result["error"],
                "error_type": error_result.get("type", "unknown"),
                "query_preview": query_preview
            }
        )

    first_result = results[0] if results else {}
    query_type = first_result.get("type", "unknown")

    response_data = None
    response_message = None
    affected_rows = None
    metadata = {}
    column_types = None

    # Construir la respuesta basada en el tipo de consulta, si es un select recuperar
    # los datos, si es insert/delete/crear tabla, devolver mensaje y filas afectadas
    if query_type == "select":
        response_data = first_result.get("data", [])
        affected_rows = first_result.get("rows_count", 0)
        table_name = first_result.get("table_name")
        metadata = {
            "table_name": table_name,
            "rows_count": first_result.get("rows_count", 0)
        }
        response_message = f"{affected_rows} filas recuperadas"
        
        # Formatear distancias como porcentaje de similitud (no distancia)
        # Mientras menor la distancia original, mayor la similitud
        # Distancia 0 = 100% similar, Distancia 1 = 0% similar
        if response_data:
            for record in response_data:
                if '_distance' in record:
                    dist = record['_distance']
                    if isinstance(dist, (int, float)):
                        # Convertir distancia a porcentaje de similitud: 100 - (dist * 100)
                        similarity_percentage = (1 - dist) * 100
                        
                        # Si la similitud es muy alta (> 99.9999%), mostrar como 100%
                        if similarity_percentage > 99.9999:
                            record['_distance'] = 100.0
                        # Si es muy baja (< 0.0001%), mostrar como 0%
                        elif similarity_percentage < 0.0001:
                            record['_distance'] = 0.0
                        else:
                            # Redondear a 4 decimales
                            record['_distance'] = round(similarity_percentage, 4)

        # Obtener tipos de columnas de la tabla para el frontend
        if table_name and table_name in get_tm().tables:
            column_types = {}
            for col in get_tm().tables[table_name].columns:
                column_types[col.name] = col.data_type.value
            print(f"[DEBUG] column_types enviados: {column_types}")

    elif query_type in ["insert", "delete", "create_table", "create_table_from_file"]:
        response_message = first_result.get("message", "Operación completada")
        affected_rows = first_result.get("deleted_count", 1 if query_type == "insert" else 0)
        metadata = {
            "table_name": first_result.get("table_name"),
            "operation": query_type
        }

    _add_to_history(request.query, execution_time_ms, success=True)

    return QueryResponse(
        success=True,
        results=results,
        query_preview=query_preview,
        data=response_data,
        message=response_message,
        query_type=query_type,
        execution_time_ms=round(execution_time_ms, 2),
        affected_rows=affected_rows,
        metadata=metadata,
        column_types=column_types
    )

# Añade a la lista alguna accion o consulta realizada, que puede o no haber sido exitosa
def _add_to_history(query: str, execution_time_ms: float, success: bool = True):
    global query_history

    history_item = QueryHistoryItem(
        query=query,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        is_active=False,
        execution_time_ms=round(execution_time_ms, 2),
        success=success
    )

    query_history.insert(0, history_item)

    if len(query_history) > 50:
        query_history = query_history[:50]


# ============================================================================
# ENDPOINTS PARA ARCHIVOS MULTIMEDIA
# ============================================================================

@router.get("/files")
async def serve_file(path: str):
    """Sirve un archivo desde el sistema de archivos"""
    # Normalizar el path
    normalized_path = path.replace('\\', '/')

    # Validar que el path no intente escapar del directorio de datos
    if '..' in normalized_path:
        raise HTTPException(status_code=400, detail="Path inválido")

    # Obtener el directorio raíz del proyecto (un nivel arriba de api/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Buscar el archivo en diferentes ubicaciones
    possible_paths = [
        normalized_path,  # Ruta relativa desde donde corre el servidor
        os.path.join("data", normalized_path),  # Relativa desde api/
        os.path.join(UPLOADS_DIR, os.path.basename(normalized_path)),  # Uploads
        os.path.join(project_root, normalized_path),  # Ruta desde project root
        os.path.join(project_root, "data", normalized_path)  # data/ desde project root
    ]

    file_path = None
    for p in possible_paths:
        if os.path.exists(p) and os.path.isfile(p):
            file_path = p
            break

    if not file_path:
        raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {path}")

    # Determinar el mime type
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "application/octet-stream"

    return FileResponse(
        path=file_path,
        media_type=mime_type,
        filename=os.path.basename(file_path)
    )


@router.get("/uploads", response_model=UploadedFilesResponse)
async def list_uploads():
    """Lista todos los archivos subidos"""
    files_data = _load_uploads_registry()
    files = [UploadedFile(**f) for f in files_data]
    return UploadedFilesResponse(files=files)


@router.post("/uploads", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Sube un archivo para usar en búsquedas KNN"""
    # Validar tipo de archivo
    file_type = _get_file_type(file.filename)
    if file_type == "UNKNOWN":
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado. Permitidos: {ALLOWED_IMAGE_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS}"
        )

    # Generar ID único y nombre de archivo
    file_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1].lower()
    safe_filename = f"{file_id}{ext}"
    file_path = os.path.join(UPLOADS_DIR, safe_filename)

    # Guardar el archivo
    try:
        contents = await file.read()
        with open(file_path, 'wb') as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar archivo: {str(e)}")

    # Crear registro del archivo
    uploaded_file = {
        "id": file_id,
        "name": file.filename,
        "path": file_path.replace('\\', '/'),
        "type": file_type,
        "size": len(contents),
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "preview_url": f"/files?path={file_path.replace(chr(92), '/')}"
    }

    # Agregar al registro
    files_data = _load_uploads_registry()
    files_data.append(uploaded_file)
    _save_uploads_registry(files_data)

    return UploadResponse(
        success=True,
        file=UploadedFile(**uploaded_file),
        message=f"Archivo '{file.filename}' subido exitosamente"
    )


@router.delete("/uploads/{file_id}")
async def delete_upload(file_id: str):
    """Elimina un archivo subido"""
    files_data = _load_uploads_registry()

    # Buscar el archivo
    file_to_delete = None
    for f in files_data:
        if f["id"] == file_id:
            file_to_delete = f
            break

    if not file_to_delete:
        raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {file_id}")

    # Eliminar el archivo físico
    try:
        if os.path.exists(file_to_delete["path"]):
            os.remove(file_to_delete["path"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al eliminar archivo: {str(e)}")

    # Eliminar del registro
    files_data = [f for f in files_data if f["id"] != file_id]
    _save_uploads_registry(files_data)

    return {"success": True, "message": f"Archivo '{file_to_delete['name']}' eliminado"}
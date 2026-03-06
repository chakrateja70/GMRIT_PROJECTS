import os
from fastapi import APIRouter, UploadFile, File
from src.schemas.response import QuerySuccessResponse
from src.schemas.response import QueryRequest
from src.utils.swagger import uploadendpoint, queryendpoint
from src.db.upload import process_uploaded_files

from src.services.rag_service import get_rag_response

router = APIRouter()

# --- File Upload Endpoint ---
@router.post("/upload", **uploadendpoint)
async def upload_files(uploaded_files: list[UploadFile] = File(...)):
    """
    Upload and process files, returning processing result.
    """
    result = process_uploaded_files(uploaded_files)
    return result

# --- Query Endpoint ---
@router.post("/query", response_model=QuerySuccessResponse, responses=queryendpoint["responses"])
async def query_rag_service(request: QueryRequest):
    """
    Query the RAG service and return the response.
    """
    response = get_rag_response(
        query=request.query,
        top_k=request.top_k,
        min_score=request.min_score
    )
    return response
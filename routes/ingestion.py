
import shutil
import traceback
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
from langchain_huggingface import HuggingFaceEmbeddings

from vectorstore.chroma_vectorstore import ChromaVectorStore
from ingestion.ingest_documents import ingest_document

router = APIRouter()


@router.post("/upload")
def upload_document(file: UploadFile = File(...)):

    try:
        # Check file type
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed."
            )

        # Create upload directory
        upload_dir = Path("data/raw_pdfs")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Safely get filename
        safe_filename = Path(file.filename).name
        file_path = upload_dir / safe_filename

        # Save uploaded PDF
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Initialize local Hugging Face embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Initialize local ChromaDB
        vector_store = ChromaVectorStore(
            embeddings=embeddings,
            persist_directory="data/chroma_db"
        )

        # Process and ingest uploaded document
        ingest_document(
            pdf_path=str(file_path),
            embeddings=embeddings,
            vector_store=vector_store
        )

        return {
            "message": "Document uploaded and processed successfully",
            "file_name": safe_filename
        }

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {str(e)}"
        )

    finally:
        file.file.close()

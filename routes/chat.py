
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from vectorstore.chroma_vectorstore import ChromaVectorStore
from rag.kpi_extractor_rag import Retriever
from llm.ollama_client import OllamaClient

load_dotenv()

router = APIRouter()

# Initialize existing services
vector_store = ChromaVectorStore(
    persist_directory="data/chroma_db"
)

retriever = Retriever(vector_store.client)

ollama = OllamaClient()


class ChatRequest(BaseModel):
    question: str
    company: str | None = None
    year: int | None = None


@router.post("/chat")
async def chat(request: ChatRequest):

    try:
        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        # Retrieve relevant chunks.
        # Temporarily avoid year filtering because the stored
        # ChromaDB metadata year may be 2024, while the report
        # contains FY2025 financial information.
        docs = retriever.invoke(
            query=question,
            company=request.company,
            year=None,
            top_k=3
        )

        if not docs:
            return {
                "answer": (
                    "I couldn't find relevant information in "
                    "the uploaded financial reports."
                ),
                "retrieved_chunks": 0
            }

        context = "\n\n".join(
            doc.page_content for doc in docs
        )
        prompt = f"""
        Answer using only the financial report context.
        Be accurate, concise, and do not invent figures.
        If information is missing, say so.
        Context:
        {context}
        Question:
        {question}
        Answer:
        """

        # Generate response using existing OllamaClient
        answer = ollama.generate_response(prompt)

        return {
            "answer": answer,
            "company": request.company,
            "year": request.year,
            "retrieved_chunks": len(docs)
        }

    except HTTPException:
        raise

    except Exception as e:
        print(f"Chat processing error: {e}", flush=True)

        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )

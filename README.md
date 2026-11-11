# AI-Powered Investor Intelligence Platform

An AI-powered financial intelligence platform that processes company annual reports, extracts key financial metrics, enables semantic search, and provides context-aware answers through a Retrieval-Augmented Generation (RAG) chatbot.

The platform combines document ingestion, semantic chunking, vector search, Large Language Models (LLMs), and PostgreSQL to simplify financial research and investment analysis
<img width="1897" height="847" alt="Screenshot 2026-11-11 194305" src="https://github.com/user-attachments/assets/32510161-dcc2-4462-a586-3a79bf3731fa" />

## Prerequisites

- Python 3.10+
- Ollama
- PostgreSQL
- Git (optional, for local version control)

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sanjanajames31/investor-intelligence-platform.git
cd investor-intelligence-platform
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Ollama

Install and start Ollama, then download the model:

```bash
ollama pull llama3.2:3b
```

Ensure Ollama is running before using the chatbot.

### 6. Configure Environment Variables

Create a `.env` file in the project root and configure the required PostgreSQL connection settings and other environment variables.

**Important:** Never upload your `.env` file or expose database credentials in the repository.

### 7. Set Up PostgreSQL

Create the PostgreSQL database and initialize the required tables using the database setup scripts included in the project.

### 8. Run the Application

Start the FastAPI application:

```bash
python -m uvicorn main:app --reload
```

Open the application in your browser:

```text
http://127.0.0.1:8000/
```

To access the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Project Features

- **Annual Report Processing:** Upload and process company annual reports in PDF format.
- **PDF-to-Markdown Conversion:** Convert PDF documents into structured Markdown content.
- **Semantic Chunking:** Divide financial documents into meaningful text chunks.
- **Vector Database Integration:** Store and retrieve document embeddings using ChromaDB.
- **Financial KPI Extraction:** Extract important financial metrics, including revenue, net income, operating income, cash flow, assets, and liabilities.
- **RAG-Based Chatbot:** Ask natural-language questions and receive answers grounded in retrieved financial documents.
- **LLM Integration:** Generate financial insights and responses using Ollama and Llama 3.2.
- **PostgreSQL Storage:** Store extracted financial metrics in a relational database.
- **Investor Dashboard:** View financial metrics and insights through a web-based dashboard.
- **REST API:** Access application functionality through FastAPI endpoints.

## Technology Stack

### Backend
- Python
- FastAPI
- Uvicorn

### AI & Language Models
- Ollama
- Llama 3.2 (3B)
- LangChain

### Document Processing
- PyMuPDF
- PyMuPDF4LLM
- Semantic Chunking
- Hugging Face Embeddings

### Vector Database
- ChromaDB

### Relational Database
- PostgreSQL

### Frontend
- HTML
- CSS
- JavaScript
- Jinja2 Templates

### Environment & Dependencies
- Python Virtual Environment
- pip
- python-dotenv

## Application Workflow

1. Upload a company's annual report in PDF format.
2. Convert the PDF into Markdown.
3. Split the document into semantically meaningful chunks.
4. Generate embeddings and store the chunks in ChromaDB.
5. Extract financial KPIs and generate insights.
6. Store structured financial metrics in PostgreSQL.
7. Retrieve relevant document chunks when a user asks a question.
8. Generate context-aware responses using the Ollama LLM.
9. Display financial information through the investor dashboard.

## Notes

- Ensure PostgreSQL is installed and running before starting the application.
- Ensure Ollama and the required language model are available locally.
- Configure all required environment variables before running the application.
- Keep uploaded reports, generated files, local databases, and environment secrets out of source control.
- The application uses local ChromaDB storage for document embeddings.
- For production deployment, consider containerization, cloud-hosted databases, authentication, and secure secrets management.

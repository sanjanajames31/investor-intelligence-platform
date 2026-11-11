
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import text

from routes.dashboard import router as dashboard_router
from routes.health import router as health_router
from routes.ingestion import router as ingestion_router
from routes.chat import router as chat_router
from database.postgres_sql import get_engine

app = FastAPI(
    title="Investor Intelligence API",
    version="1.0.0"
)

# Connect frontend folders
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(directory="template")


# Existing API routes
app.include_router(
    health_router,
    tags=["Health"]
)

app.include_router(
    dashboard_router,
    prefix="/api",
    tags=["Dashboard"]
)

app.include_router(
    ingestion_router,
    prefix="/api",
    tags=["Ingestion"]
)

app.include_router(
    chat_router,
    prefix="/api",
    tags=["Chat"]
)


# Homepage
@app.get("/")
def home(request: Request):

    engine = get_engine()

    with engine.connect() as connection:

        result = connection.execute(
            text("SELECT * FROM financial_metrics ORDER BY company")
        )

        metrics = [dict(row._mapping) for row in result]

        total_companies = len(set(
            row["company"] for row in metrics if row["company"]
        ))

        total_reports = len(metrics)

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "metrics": metrics,
            "total_companies": total_companies,
            "total_reports": total_reports
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

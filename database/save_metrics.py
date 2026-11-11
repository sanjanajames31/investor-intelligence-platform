
import json

from sqlalchemy import text
from database.postgres_sql import get_engine


def convert_value(value):
    """Convert extracted values into database-compatible strings."""

    if value is None:
        return None

    # Convert empty or null-like strings into actual SQL NULL.
    if isinstance(value, str):
        cleaned_value = value.strip()

        if cleaned_value.lower() in {"", "null", "none", "nan"}:
            return None

        return cleaned_value

    if isinstance(value, (list, dict)):
        return json.dumps(value)

    return str(value)


def save_metrics(company: str, year: int, metrics: dict) -> None:
    """Insert or update financial metrics for a company and year."""

    if not company:
        raise ValueError("A valid company name is required.")

    if year is None:
        raise ValueError("A valid financial year is required.")

    values = {
        "company": company.lower().strip(),

        # PostgreSQL year column is VARCHAR.
        "year": str(year),

        "revenue": convert_value(metrics.get("revenue")),
        "net_income": convert_value(metrics.get("net_income")),
        "operating_income": convert_value(metrics.get("operating_income")),
        "cash_flow": convert_value(metrics.get("cash_flow")),
        "total_assets": convert_value(metrics.get("total_assets")),
        "total_liabilities": convert_value(metrics.get("total_liabilities")),
        "risk_factors": convert_value(metrics.get("risk_factors")),
        "growth_drivers": convert_value(metrics.get("growth_drivers"))
    }

    update_query = text("""
        UPDATE financial_metrics
        SET
            revenue = :revenue,
            net_income = :net_income,
            operating_income = :operating_income,
            cash_flow = :cash_flow,
            total_assets = :total_assets,
            total_liabilities = :total_liabilities,
            risk_factors = :risk_factors,
            growth_drivers = :growth_drivers
        WHERE LOWER(company) = :company
          AND year = :year
    """)

    insert_query = text("""
        INSERT INTO financial_metrics (
            company,
            year,
            revenue,
            net_income,
            operating_income,
            cash_flow,
            total_assets,
            total_liabilities,
            risk_factors,
            growth_drivers
        )
        VALUES (
            :company,
            :year,
            :revenue,
            :net_income,
            :operating_income,
            :cash_flow,
            :total_assets,
            :total_liabilities,
            :risk_factors,
            :growth_drivers
        )
    """)

    engine = get_engine()

    with engine.begin() as connection:

        result = connection.execute(update_query, values)

        if result.rowcount == 0:
            connection.execute(insert_query, values)

            print(
                f"New record inserted: {company} ({year})",
                flush=True
            )

        else:
            print(
                f"Existing record updated: {company} ({year})",
                flush=True
            )


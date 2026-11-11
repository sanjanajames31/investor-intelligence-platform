
import os

import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

load_dotenv(override=True)


def get_database_config():
    """Read and validate PostgreSQL configuration."""

    required = [
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DATABASE"
    ]

    config = {}

    for key in required:
        value = os.getenv(key)

        if not value:
            raise ValueError(
                f"Missing {key} in your .env file."
            )

        config[key] = value

    config["POSTGRES_PORT"] = int(config["POSTGRES_PORT"])
    config["POSTGRES_SSLMODE"] = os.getenv(
        "POSTGRES_SSLMODE", "prefer"
    )

    return config


def get_engine(database=None):
    """Create a SQLAlchemy PostgreSQL engine."""

    config = get_database_config()

    target_database = database or config["POSTGRES_DATABASE"]

    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=config["POSTGRES_USER"],
        password=config["POSTGRES_PASSWORD"],
        host=config["POSTGRES_HOST"],
        port=config["POSTGRES_PORT"],
        database=target_database,
        query={
            "sslmode": config["POSTGRES_SSLMODE"]
        }
    )

    return create_engine(
        connection_url,
        pool_pre_ping=True
    )


def create_database():
    """Create the target PostgreSQL database if needed."""

    config = get_database_config()
    target_db = config["POSTGRES_DATABASE"]

    conn = None

    try:
        conn = psycopg2.connect(
            host=config["POSTGRES_HOST"],
            port=config["POSTGRES_PORT"],
            database="postgres",
            user=config["POSTGRES_USER"],
            password=config["POSTGRES_PASSWORD"],
            sslmode=config["POSTGRES_SSLMODE"]
        )

        conn.autocommit = True

        with conn.cursor() as cursor:

            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (target_db,)
            )

            if cursor.fetchone():
                print(f"Database '{target_db}' already exists.")

            else:
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(target_db)
                    )
                )

                print(
                    f"Database '{target_db}' created successfully."
                )

    except Exception as exc:
        print(f"Database creation failed: {exc}")
        raise

    finally:
        if conn is not None:
            conn.close()


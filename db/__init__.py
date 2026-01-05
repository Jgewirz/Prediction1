"""
Database module for Kalshi Deep Trading Bot.

Provides SQLite and PostgreSQL storage for betting decisions, outcomes, and performance tracking.
"""

from .database import Database, get_database
from .postgres import PostgresDatabase, get_postgres_database, close_postgres_database
from .queries import Queries

__all__ = [
    "Database",
    "get_database",
    "PostgresDatabase",
    "get_postgres_database",
    "close_postgres_database",
    "Queries"
]

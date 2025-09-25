"""
Database connection and query utilities for MortgageAI.

This module provides async database connections using asyncpg
and utilities for managing database operations.
"""

import asyncpg
import logging
from typing import Optional, Any, Dict, List
from contextlib import asynccontextmanager

from .config import settings


class DatabaseManager:
    """Manages database connections and operations."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize the database connection pool."""
        try:
            db_config = settings.get_database_config()

            self.pool = await asyncpg.create_pool(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                ssl=db_config['ssl'],
                min_size=5,
                max_size=20,
                command_timeout=60
            )

            self.logger.info("Database connection pool initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise

    async def close(self):
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
            self.logger.info("Database connection pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if not self.pool:
            raise RuntimeError("Database not initialized")

        conn = await self.pool.acquire()
        try:
            yield conn
        finally:
            await self.pool.release(conn)


# Global database manager instance
db_manager = DatabaseManager()


async def get_db_connection():
    """
    Get a database connection.

    This is a convenience function for getting connections in handlers.
    """
    async with db_manager.get_connection() as conn:
        return conn


async def initialize_database():
    """Initialize the database and create tables if they don't exist."""
    try:
        async with db_manager.get_connection() as conn:
            # Run the schema.sql file
            with open('schema.sql', 'r') as f:
                schema_sql = f.read()

            await conn.execute(schema_sql)
            logging.info("Database schema initialized")

    except Exception as e:
        logging.error(f"Failed to initialize database schema: {str(e)}")
        raise


async def execute_query(query: str, *args) -> List[asyncpg.Record]:
    """
    Execute a SELECT query and return results.

    Args:
        query: SQL query string
        *args: Query parameters

    Returns:
        List of result records
    """
    async with db_manager.get_connection() as conn:
        return await conn.fetch(query, *args)


async def execute_mutation(query: str, *args) -> str:
    """
    Execute an INSERT, UPDATE, or DELETE query.

    Args:
        query: SQL query string
        *args: Query parameters

    Returns:
        Status message
    """
    async with db_manager.get_connection() as conn:
        result = await conn.execute(query, *args)
        return result


async def execute_scalar(query: str, *args) -> Any:
    """
    Execute a query that returns a single value.

    Args:
        query: SQL query string
        *args: Query parameters

    Returns:
        Single value result
    """
    async with db_manager.get_connection() as conn:
        return await conn.fetchval(query, *args)


async def log_agent_interaction(
    application_id: str,
    agent_type: str,
    interaction_type: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    processing_time_ms: int,
    success: bool = True,
    error_message: Optional[str] = None
):
    """
    Log an agent interaction for audit purposes.

    Args:
        application_id: UUID of the mortgage application
        agent_type: Type of agent ('compliance' or 'quality_control')
        interaction_type: Type of interaction
        input_data: Input data provided to the agent
        output_data: Output data from the agent
        processing_time_ms: Processing time in milliseconds
        success: Whether the interaction was successful
        error_message: Error message if unsuccessful
    """
    try:
        await execute_mutation("""
            INSERT INTO agent_interactions (
                application_id, agent_type, interaction_type,
                input_data, output_data, processing_time_ms,
                success, error_message
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        application_id, agent_type, interaction_type,
        input_data, output_data, processing_time_ms,
        success, error_message
        )

    except Exception as e:
        logging.error(f"Failed to log agent interaction: {str(e)}")


async def create_mortgage_application(
    user_id: Optional[str],
    application_data: Dict[str, Any]
) -> str:
    """
    Create a new mortgage application.

    Args:
        user_id: User ID if authenticated
        application_data: Application data

    Returns:
        Application ID
    """
    try:
        application_id = await execute_scalar("""
            INSERT INTO mortgage_applications (
                user_id, application_number, applicant_data
            ) VALUES ($1, $2, $3)
            RETURNING id
        """,
        user_id,
        f"APP-{str(asyncio.secure_random().hex()[:8]).upper()}",
        application_data
        )

        return str(application_id)

    except Exception as e:
        logging.error(f"Failed to create mortgage application: {str(e)}")
        raise


async def update_application_status(
    application_id: str,
    status: str,
    additional_data: Optional[Dict[str, Any]] = None
):
    """
    Update mortgage application status.

    Args:
        application_id: Application ID
        status: New status
        additional_data: Additional data to update
    """
    try:
        update_fields = ["status = $2"]
        params = [application_id, status]
        param_count = 2

        if additional_data:
            for key, value in additional_data.items():
                param_count += 1
                update_fields.append(f"{key} = ${param_count}")
                params.append(value)

        query = f"""
            UPDATE mortgage_applications
            SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
        """

        await execute_mutation(query, *params)

    except Exception as e:
        logging.error(f"Failed to update application status: {str(e)}")
        raise

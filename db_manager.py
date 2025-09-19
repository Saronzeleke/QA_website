import aiomysql
import redis.asyncio as redis
import json
import logging
from datetime import datetime
from setting import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.pool = None

    async def connect(self):
        try:
            self.pool = await aiomysql.create_pool(
                host=settings.mysql_host,
                port=settings.mysql_port,
                user=settings.mysql_user,
                password=settings.mysql_password,
                db=settings.mysql_database,
                autocommit=True
            )
            logger.info("MySQL connection pool created")
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {e}")
            raise

    async def close(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            logger.info("MySQL connection pool closed")

async def _check_columns(pool: aiomysql.Pool, table: str, columns: list) -> list:
    """Check which columns exist in the table."""
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"SHOW COLUMNS FROM {table}")
                existing_columns = [row[0] for row in await cursor.fetchall()]
                return [col for col in columns if col in existing_columns]
    except Exception as e:
        logger.error(f"Error checking columns for {table}: {e}")
        return []

async def query_all_db_content(pool: aiomysql.Pool, last_extraction: datetime = None) -> list:
    tables = [
        {
            "table": "bravo_airline",
            "id_col": "id",
            "title_col": "name",
            "content_col": "name",  
            "filter_deleted": True,
            "extra_fields": []
        },
     {
    "table": "bravo_spaces",
    "id_col": "id",
    "title_col": "title",
    "content_col": "content",
    "filter_deleted": True,
    "extra_fields": ["description", "address", "surrounding"]
},
        {
            "table": "bravo_boats",
            "id_col": "id",
            "title_col": "title",
            "content_col": "content",
            "filter_deleted": True,
            "extra_fields": ["description"]
        },
        {
            "table": "bravo_airport",
            "id_col": "id",
            "title_col": "name",
            "content_col": "name",  
            "filter_deleted": True,
            "extra_fields": []
        },
        {
            "table": "bravo_tours",
            "id_col": "id",
            "title_col": "title",
            "content_col": "content",
            "filter_deleted": True,
            "extra_fields": []
        },
        {
            "table": "bravo_hotels",
            "id_col": "id",
            "title_col": "title",
            "content_col": "content",
            "filter_deleted": True,
            "extra_fields": ["description"]
        },
        {
            "table": "bravo_locations",
            "id_col": "id",
            "title_col": "name",
            "content_col": "name", 
            "filter_deleted": True,
            "extra_fields": []
        },
        {
            "table": "core_pages",
            "id_col": "id",
            "title_col": "title",
            "content_col": "content",
            "filter_deleted": False,
            "extra_fields": []
        },
        {
            "table": "core_news",
            "id_col": "id",
            "title_col": "title",
            "content_col": "content",
            "filter_deleted": True,
            "extra_fields": []
        },
    ]

    results = []

    for table_config in tables:
        table = table_config["table"]
        id_col = table_config["id_col"]
        title_col = table_config["title_col"]
        content_col = table_config["content_col"]
        filter_deleted = table_config["filter_deleted"]
        extra_fields = table_config["extra_fields"]

        try:
            # Check existing columns
            check_cols = [id_col, title_col, content_col] + extra_fields + ["status", "deleted_at", "updated_at", "created_at"]
            existing_cols = await _check_columns(pool, table, check_cols)

            # Build SELECT clause
            cols = [f"{id_col} AS id", f"{title_col} AS title", f"{content_col} AS content"]
            valid_extra_fields = [f for f in extra_fields if f in existing_cols]
            cols.extend([f"COALESCE({extra}, '') AS {extra}" for extra in valid_extra_fields])

            # Add updated_at and created_at if they exist and are not already included
            if "updated_at" in existing_cols and "updated_at" not in [title_col, content_col] + valid_extra_fields:
                cols.append("updated_at")
            if "created_at" in existing_cols and "created_at" not in [title_col, content_col] + valid_extra_fields:
                cols.append("created_at")

            select_clause = ", ".join(cols)
            where_clause = []
            params = []

            if "status" in existing_cols:
                where_clause.append("status = 'publish'")
            if filter_deleted and "deleted_at" in existing_cols:
                where_clause.append("deleted_at IS NULL")
            if last_extraction and "updated_at" in existing_cols:
                where_clause.append("updated_at > %s")
                params.append(last_extraction)

            query = f"SELECT {select_clause} FROM {table}"
            if where_clause:
                query += f" WHERE {' AND '.join(where_clause)}"

            logger.debug(f"Querying {table}: {query}")

            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(query, params)
                    rows = await cursor.fetchall()

                    for row in rows:
                        title = str(row.get("title", "") or "").strip()
                        content = str(row.get("content", "") or "").strip()

                        for extra in valid_extra_fields:
                            extra_val = str(row.get(extra, "") or "").strip()
                            if extra_val:
                                content += " " + extra_val
                        content = content.strip()

                        if not content or len(content) < 20:
                            continue

                        updated = row.get("updated_at") or row.get("created_at") or datetime.now()
                        updated_iso = updated.isoformat() if isinstance(updated, datetime) else str(updated)

                        results.append({
                            "id": row["id"],
                            "title": title,
                            "content": content,
                            "source": table,
                            "updated_at": updated_iso
                        })

        except Exception as e:
            logger.error(f"Error querying table '{table}': {e}")
            continue

    logger.info(f"Loaded {len(results)} records from MySQL knowledge base")
    return results

async def query_db_content(pool: aiomysql.Pool, query: str, redis_client: redis.Redis = None) -> list:
    cache_key = f"db_query:{query}"
    results = []

    if redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Redis get error: {e}")

    tables = [
        {
            "table": "bravo_boats",
            "id_col": "id",
            "title_col": "title",
            "content_col": "content",
            "filter_deleted": True,
            "extra_fields": ["description"]
        },
        {
    "table": "bravo_spaces",
    "id_col": "id",
    "title_col": "title",
    "content_col": "content",
    "filter_deleted": True,
    "extra_fields": ["description", "address", "surrounding"]
},
        {
            "table": "bravo_airport",
            "id_col": "id",
            "title_col": "name",
            "content_col": "name",
            "filter_deleted": True,
            "extra_fields": []
        },
        {
            "table": "bravo_airline",
            "id_col": "id",
            "title_col": "name",
            "content_col": "name",
            "filter_deleted": True,
            "extra_fields": []
        },
        {
            "table": "bravo_tours",
            "id_col": "id",
            "title_col": "title",
            "content_col": "content",
            "filter_deleted": True,
            "extra_fields": []
        },
        {
            "table": "bravo_hotels",
            "id_col": "id",
            "title_col": "title",
            "content_col": "content",
            "filter_deleted": True,
            "extra_fields": ["description"]
        },
        {
            "table": "bravo_locations",
            "id_col": "id",
            "title_col": "name",
            "content_col": "name",
            "filter_deleted": True,
            "extra_fields": []
        },
        {
            "table": "core_pages",
            "id_col": "id",
            "title_col": "title",
            "content_col": "content",
            "filter_deleted": False,
            "extra_fields": []
        },
        {
            "table": "core_news",
            "id_col": "id",
            "title_col": "title",
            "content_col": "content",
            "filter_deleted": True,
            "extra_fields": []
        },
    ]

    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            for table_config in tables:
                table = table_config["table"]
                title_col = table_config["title_col"]
                content_col = table_config["content_col"]
                extra_fields = table_config["extra_fields"]
                filter_deleted = table_config["filter_deleted"]

                try:
                    # Check existing columns
                    check_cols = [title_col, content_col] + extra_fields + ["status", "deleted_at", "updated_at", "created_at"]
                    existing_cols = await _check_columns(pool, table, check_cols)
                    valid_extra_fields = [f for f in extra_fields if f in existing_cols]

                    fulltext_cols = [title_col, content_col] + valid_extra_fields
                    fulltext_cols_str = ", ".join(fulltext_cols)
                    select_fields = [f"id", f"{title_col} AS title", f"{content_col} AS content"]
                    select_fields.extend([f"COALESCE({f}, '') AS {f}" for f in valid_extra_fields])
                    if "updated_at" in existing_cols:
                        select_fields.append("updated_at")

                    sql_fulltext = f"SELECT {', '.join(select_fields)} FROM {table}"

                    where_clause = []
                    params = []

                    if "status" in existing_cols:
                        where_clause.append("status = 'publish'")
                    if filter_deleted and "deleted_at" in existing_cols:
                        where_clause.append("deleted_at IS NULL")
                    where_clause.append(f"MATCH({fulltext_cols_str}) AGAINST (%s IN BOOLEAN MODE)")
                    params.append(query)

                    sql_fulltext += f" WHERE {' AND '.join(where_clause)}"

                    try:
                        await cursor.execute(sql_fulltext, params)
                        rows = await cursor.fetchall()
                    except Exception:
                        rows = []

                    if not rows:
                        like_q = f"%{query.replace('%','').replace('_','')}%"
                        like_sql = f"SELECT {', '.join(select_fields)} FROM {table}"
                        like_params = [like_q] * (2 + len(valid_extra_fields))
                        like_where = [f"{title_col} LIKE %s", f"{content_col} LIKE %s"]
                        like_where.extend([f"{f} LIKE %s" for f in valid_extra_fields])
                        like_where_clause = f"({' OR '.join(like_where)})"
                        if "status" in existing_cols:
                            like_where_clause = f"status = 'publish' AND ({like_where_clause})"
                        if filter_deleted and "deleted_at" in existing_cols:
                            like_where_clause += " AND deleted_at IS NULL"
                        like_sql += f" WHERE {like_where_clause}"
                        await cursor.execute(like_sql, like_params)
                        rows = await cursor.fetchall()

                    for row in rows:
                        title = str(row.get("title", "") or "").strip()
                        content = str(row.get("content", "") or "").strip()

                        for extra in valid_extra_fields:
                            extra_val = str(row.get(extra, "") or "").strip()
                            if extra_val:
                                content += " " + extra_val
                        content = content.strip()

                        if not content or len(content) < 10:
                            continue

                        updated = row.get("updated_at") or row.get("created_at") or datetime.now()
                        updated_iso = updated.isoformat() if isinstance(updated, datetime) else str(updated)

                        results.append({
                            "id": row["id"],
                            "title": title,
                            "content": content,
                            "source": table,
                            "updated_at": updated_iso
                        })

                except Exception as e:
                    logger.error(f"Error querying table '{table}': {e}")
                    continue

    if redis_client and results:
        try:
            await redis_client.setex(cache_key, settings.redis_cache_ttl, json.dumps(results))
        except Exception as e:
            logger.error(f"Redis setex error: {e}")

    return results
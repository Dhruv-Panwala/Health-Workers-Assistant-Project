import argparse
import argparse
import re
import shutil
import sqlite3
from pathlib import Path


FACT_TABLE_NAME = "assistant_fact_values"
MONTHLY_SUMMARY_TABLE_NAME = "assistant_monthly_summary"
EXPLAINABLE_CACHE_TABLE_NAME = "assistant_explainable_cache"
METRIC_LOOKUP_TABLE_NAME = "assistant_metric_lookup"
ORGUNIT_LOOKUP_TABLE_NAME = "assistant_orgunit_lookup"
RUNTIME_METADATA_TABLE_NAME = "assistant_runtime_metadata"


def sqlite_regexp(pattern: str, value) -> int:
    if value is None:
        return 0
    try:
        return 1 if re.search(pattern, str(value)) else 0
    except re.error:
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build optimized SQLite tables for the Android runtime.")
    parser.add_argument("--source", required=True, help="Path to the source dhis2.sqlite database.")
    parser.add_argument("--target", required=True, help="Path to write the optimized database copy.")
    parser.add_argument(
        "--partial-fact-start-date",
        help="Optional ISO date. When provided, build assistant_fact_values only for periods starting on or after this date.",
    )
    return parser.parse_args()


def copy_database(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    shutil.copy2(source_path, target_path)


def optimize_database(target_path: Path, partial_fact_start_date: str | None = None) -> None:
    connection = sqlite3.connect(str(target_path))
    try:
        connection.create_function("REGEXP", 2, sqlite_regexp)
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute("PRAGMA cache_size = -200000")

        connection.executescript(
            f"""
            DROP TABLE IF EXISTS {RUNTIME_METADATA_TABLE_NAME};
            DROP TABLE IF EXISTS {ORGUNIT_LOOKUP_TABLE_NAME};
            DROP TABLE IF EXISTS {METRIC_LOOKUP_TABLE_NAME};
            DROP TABLE IF EXISTS {EXPLAINABLE_CACHE_TABLE_NAME};
            DROP TABLE IF EXISTS {MONTHLY_SUMMARY_TABLE_NAME};
            DROP TABLE IF EXISTS {FACT_TABLE_NAME};

            CREATE TABLE {RUNTIME_METADATA_TABLE_NAME} (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE {ORGUNIT_LOOKUP_TABLE_NAME} AS
            SELECT
                organisationunitid,
                TRIM(name) AS name,
                lower(TRIM(name)) AS normalized_name
            FROM organisationunit
            WHERE trim(coalesce(name, '')) <> '';

            CREATE INDEX idx_assistant_orgunit_lookup_normalized_name
                ON {ORGUNIT_LOOKUP_TABLE_NAME}(normalized_name);

            CREATE TABLE {METRIC_LOOKUP_TABLE_NAME} AS
            SELECT
                dataelementid,
                TRIM(name) AS name,
                lower(TRIM(name)) AS normalized_name,
                TRIM(COALESCE(shortname, '')) AS shortname,
                lower(TRIM(COALESCE(shortname, ''))) AS normalized_shortname,
                TRIM(COALESCE(code, '')) AS code,
                lower(TRIM(COALESCE(code, ''))) AS normalized_code,
                TRIM(COALESCE(description, '')) AS description,
                lower(
                    trim(
                        coalesce(name, '') || ' ' ||
                        coalesce(shortname, '') || ' ' ||
                        coalesce(code, '') || ' ' ||
                        coalesce(description, '')
                    )
                ) AS search_blob
            FROM dataelement;

            CREATE INDEX idx_assistant_metric_lookup_normalized_name
                ON {METRIC_LOOKUP_TABLE_NAME}(normalized_name);
            CREATE INDEX idx_assistant_metric_lookup_search_blob
                ON {METRIC_LOOKUP_TABLE_NAME}(search_blob);

            CREATE TABLE {EXPLAINABLE_CACHE_TABLE_NAME} AS
            WITH typed_values AS (
                SELECT
                    dv.dataelementid,
                    CASE
                        WHEN REGEXP('^[-]?\\d+(\\.\\d+)?$', TRIM(CAST(dv.value AS TEXT)))
                        THEN CAST(dv.value AS REAL)
                    END AS numeric_value,
                    NULLIF(TRIM(CAST(dv.value AS TEXT)), '') AS raw_text_value,
                    NULLIF(TRIM(COALESCE(dv.comment, '')), '') AS comment_value,
                    p.startdate,
                    p.enddate,
                    dv.lastupdated,
                    dv.created,
                    ROW_NUMBER() OVER (
                        PARTITION BY dv.dataelementid
                        ORDER BY p.startdate DESC, p.enddate DESC, dv.lastupdated DESC, dv.created DESC
                    ) AS recency_rank
                FROM datavalue AS dv
                JOIN period AS p ON dv.periodid = p.periodid
            ),
            numeric_rollup AS (
                SELECT
                    dataelementid,
                    COUNT(numeric_value) AS data_points,
                    COALESCE(SUM(numeric_value), 0) AS total_value,
                    MAX(CASE WHEN recency_rank = 1 THEN numeric_value END) AS latest_value,
                    MAX(CASE WHEN recency_rank = 1 THEN startdate END) AS latest_period_start,
                    MAX(CASE WHEN recency_rank = 1 THEN enddate END) AS latest_period_end,
                    MIN(startdate) AS first_period_start,
                    MAX(enddate) AS last_period_end
                FROM typed_values
                GROUP BY dataelementid
            ),
            text_rollup AS (
                SELECT
                    dataelementid,
                    MAX(CASE WHEN recency_rank = 1 THEN raw_text_value END) AS latest_text_value,
                    MAX(CASE WHEN recency_rank = 1 THEN comment_value END) AS latest_comment
                FROM typed_values
                GROUP BY dataelementid
            )
            SELECT
                de.dataelementid,
                TRIM(de.name) AS dataelement_name,
                lower(TRIM(de.name)) AS normalized_name,
                TRIM(COALESCE(de.shortname, '')) AS shortname,
                TRIM(COALESCE(de.code, '')) AS code,
                TRIM(COALESCE(de.description, '')) AS description,
                TRIM(COALESCE(de.valuetype, '')) AS valuetype,
                TRIM(COALESCE(de.aggregationtype, '')) AS aggregationtype,
                COALESCE(nr.data_points, 0) AS data_points,
                COALESCE(nr.total_value, 0) AS total_value,
                nr.latest_value,
                nr.latest_period_start,
                nr.latest_period_end,
                nr.first_period_start,
                nr.last_period_end,
                tr.latest_text_value,
                tr.latest_comment
            FROM dataelement AS de
            LEFT JOIN numeric_rollup AS nr ON nr.dataelementid = de.dataelementid
            LEFT JOIN text_rollup AS tr ON tr.dataelementid = de.dataelementid;

            CREATE INDEX idx_assistant_explainable_cache_metric
                ON {EXPLAINABLE_CACHE_TABLE_NAME}(dataelementid);
            CREATE INDEX idx_assistant_explainable_cache_normalized_name
                ON {EXPLAINABLE_CACHE_TABLE_NAME}(normalized_name);

            ANALYZE;
            """
        )

        if partial_fact_start_date:
            connection.execute(
                f"""
                CREATE TABLE {FACT_TABLE_NAME} AS
                SELECT
                    dv.dataelementid,
                    dv.sourceid,
                    dv.periodid,
                    TRIM(de.name) AS dataelement_name,
                    TRIM(ou.name) AS orgunit_name,
                    p.startdate,
                    p.enddate,
                    TRIM(COALESCE(pt.name, '')) AS period_type,
                    dv.value,
                    CASE
                        WHEN REGEXP('^[-]?\\d+(\\.\\d+)?$', TRIM(CAST(dv.value AS TEXT)))
                        THEN CAST(dv.value AS REAL)
                    END AS value_num,
                    CASE WHEN COALESCE(dv.followup, 0) <> 0 THEN 1 ELSE 0 END AS followup
                FROM datavalue AS dv
                JOIN dataelement AS de ON dv.dataelementid = de.dataelementid
                JOIN organisationunit AS ou ON dv.sourceid = ou.organisationunitid
                JOIN period AS p ON dv.periodid = p.periodid
                LEFT JOIN periodtype AS pt ON p.periodtypeid = pt.periodtypeid
                WHERE p.startdate >= ?
                """,
                (partial_fact_start_date,),
            )
            connection.execute(
                f"""
                CREATE INDEX idx_assistant_fact_values_metric_org_start
                    ON {FACT_TABLE_NAME}(dataelementid, sourceid, startdate)
                """
            )
            connection.execute(
                f"""
                CREATE INDEX idx_assistant_fact_values_org_metric_start
                    ON {FACT_TABLE_NAME}(sourceid, dataelementid, startdate)
                """
            )
            connection.execute(
                f"""
                CREATE INDEX idx_assistant_fact_values_period_type_start
                    ON {FACT_TABLE_NAME}(period_type, startdate)
                """
            )
            connection.execute(
                f"""
                CREATE INDEX idx_assistant_fact_values_followup_start
                    ON {FACT_TABLE_NAME}(followup, startdate)
                """
            )
            connection.execute(
                f"""
                CREATE INDEX idx_assistant_fact_values_value_num
                    ON {FACT_TABLE_NAME}(value_num)
                """
            )
            connection.executemany(
                f"INSERT INTO {RUNTIME_METADATA_TABLE_NAME}(key, value) VALUES (?, ?)",
                [
                    ("fact_table_mode", "partial_recent"),
                    ("fact_table_start_date", partial_fact_start_date),
                    (
                        "fact_table_end_date",
                        connection.execute(f"SELECT MAX(startdate) FROM {FACT_TABLE_NAME}").fetchone()[0] or "",
                    ),
                ],
            )
        else:
            connection.execute(
                f"INSERT INTO {RUNTIME_METADATA_TABLE_NAME}(key, value) VALUES (?, ?)",
                ("fact_table_mode", "none"),
            )
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source).expanduser().resolve()
    target_path = Path(args.target).expanduser().resolve()

    if not source_path.exists():
        raise SystemExit(f"Source database not found: {source_path}")

    copy_database(source_path, target_path)
    optimize_database(target_path, partial_fact_start_date=args.partial_fact_start_date)
    print(f"Optimized runtime database written to: {target_path}")


if __name__ == "__main__":
    main()

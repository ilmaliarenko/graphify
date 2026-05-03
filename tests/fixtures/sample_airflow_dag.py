"""Sample Airflow DAG fixture — bookstore inventory pipeline.

Fictional domain: a daily job that reads new book orders from S3, refreshes
inventory dbt models via Cosmos, and posts a daily-sales report to a webhook.
"""
from datetime import datetime, timedelta
import os
import requests

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.hooks.base import BaseHook
from airflow.datasets import Dataset
from cosmos import DbtDag, ProjectConfig, RenderConfig, ProfileConfig

WEBHOOK_CONN_ID = "bookstore_webhook"
DBT_TARGET = os.getenv("DBT_TARGET", "staging")

orders_dataset = Dataset("s3://bookstore-raw/orders/")
inventory_dataset = Dataset("s3://bookstore-raw/inventory/")

render_config = RenderConfig(
    select=["tag:bookstore", "path:models/inventory"],
    exclude=["tag:experimental", "path:models/scratch"],
)

dbt_inventory = DbtDag(
    dag_id="dbt_refresh_inventory",
    project_config=ProjectConfig("/usr/local/airflow/dbt/bookstore_dbt_project"),
    render_config=render_config,
    profile_config=ProfileConfig(profile_name="bookstore", target_name=DBT_TARGET),
    schedule=[orders_dataset, inventory_dataset],
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["bookstore", "dbt"],
)


def post_sales_report():
    """Fetch yesterday's sales summary and POST it to the partner webhook."""
    chunk_size = int(Variable.get("BOOKSTORE_REPORT_CHUNK", default_var="500"))
    region = Variable.get("BOOKSTORE_REGION")
    conn = BaseHook.get_connection(WEBHOOK_CONN_ID)
    requests.post(conn.host, json={"region": region, "chunk_size": chunk_size})


with DAG(
    dag_id="bookstore_daily_report",
    schedule="30 7 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["bookstore", "report"],
) as dag:

    pull_orders = BashOperator(
        task_id="pull_orders_from_s3",
        bash_command=(
            "set -e\n"
            "aws s3 sync s3://bookstore-raw/orders/ /tmp/orders/\n"
            f"dbt seed --target {DBT_TARGET} --select tag:bookstore_seed"
        ),
    )

    refresh_inventory = BashOperator(
        task_id="refresh_inventory_models",
        bash_command=(
            "cd /usr/local/airflow/dbt/bookstore_dbt_project\n"
            f"dbt build --target {DBT_TARGET} --select tag:bookstore --exclude tag:slow"
        ),
    )

    publish_report = PythonOperator(
        task_id="publish_daily_report",
        python_callable=post_sales_report,
        outlets=[Dataset("s3://bookstore-reports/daily/")],
    )

    notify_partners = BashOperator(
        task_id="notify_partners",
        bash_command="curl -X POST https://example.test/notify",
    )

    pull_orders >> refresh_inventory >> publish_report >> notify_partners

from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
from processing_subdag import processing_analysis_subdag
from ml_subdag import ml_analysis_subdag

# TODO: Run Airflow to fix bugs
# TODO: Move ml code to jupyter notebook
# TODO: Test docker container
# TODO: Run code on google colab to test code one time run
# TODO: Create the presentation slide of 3 - 5 pages.
# TODO: Add screenshot of airflow running to github and presentation

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 4, 26),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

parent_dag_name = 'processing_data_with_ml_analysis'
dag = DAG(
    parent_dag_name,
    default_args=default_args,
    description='ML analysis pipeline with processing subdag',
    schedule_interval=None, # disable automatic scheduling
    max_active_runs=1, # ensure only one DAG run is active at a time
    catchup=False # prevent backfilling of DAG runs
)

with dag:
    with TaskGroup('processing', tooltip='Processing Tasks') as processing:
        processing_start = DummyOperator(
            task_id='processing_start'
        )

        processing_subdag = processing_analysis_subdag(parent_dag_name, 'processing_subdag', default_args)

        processing_end = DummyOperator(
            task_id='processing_end'
        )

    with TaskGroup('ml', tooltip='ML Tasks') as ml:
        ml_start = DummyOperator(
            task_id='ml_start'
        )

        ml_subdag = ml_analysis_subdag(parent_dag_name, 'ml_subdag', default_args)

        ml_end = DummyOperator(
            task_id='end'
        )

    start = DummyOperator(
        task_id='start'
    )

    start >> processing_start >> processing_subdag >> processing_end >> ml_start >> ml_subdag >> ml_end

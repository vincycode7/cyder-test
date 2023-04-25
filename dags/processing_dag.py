from typing import List, Tuple
import pandas as pd
# from airflow.operators.papermill_operator import PapermillOperator
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os, spacy

from common_files.operators.data_operators import DataAnalyzer, load_or_download_models

# Define DAG arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 4, 25),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

processing_dag = DAG(
    'ml_analysis_pipeline',
    default_args=default_args,
    description='Analye Customer data and perform feature engineering',
    schedule_interval=timedelta(days=1),
)

download_spacy_models_task = PythonOperator(
    task_id='load_or_download_models',
    python_callable=load_or_download_models,
    dag=processing_dag
)

data_processor = DataAnalyzer(link_to_csv='inputs/UserWillChristodoulou_DataScience_TestData.csv')
analyze_data_task = PythonOperator(
    task_id='data_processor',
    python_callable=data_processor.analyze_data,
    dag=processing_dag
)

# Set task dependencies
download_spacy_models_task >> analyze_data_task # Link the tasks together
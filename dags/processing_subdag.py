from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.task_group import TaskGroup
from common_files.operators.data_operators import load_or_download_models, DataAnalyzer

def processing_analysis_subdag(parent_dag_name, child_dag_name, default_args):
    with TaskGroup(group_id='processing_analysis_tasks') as processing_analysis_tasks:
        start_task = DummyOperator(
            task_id='start',
        )

        download_spacy_models_task = PythonOperator(
            task_id='load_or_download_models',
            python_callable=load_or_download_models,
        )

        data_processor = DataAnalyzer(link_to_csv='inputs/UserWillChristodoulou_DataScience_TestData.csv')
        analyze_data_task = PythonOperator(
            task_id='data_processor',
            python_callable=data_processor.analyze_data,
        )

        # Set task dependencies
        start_task >> download_spacy_models_task >> analyze_data_task # Link the tasks together
    
    return processing_analysis_tasks
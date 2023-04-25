from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dagrun_operator import TriggerDagRunOperator
from airflow.operators.subdag_operator import SubDagOperator

# Import the tasks
from processing_dag import processing_dag
from ml_dags import ml_dag
    
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 4, 30),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'base_dag',
    default_args=default_args,
    schedule_interval=timedelta(days=1),
)

trigger_processing_dag = TriggerDagRunOperator(
    task_id='trigger_processing_dag',
    trigger_dag_id='processing_dag',
    dag=dag,
)

ml_subdag = SubDagOperator(
    task_id='ml_subdag',
    subdag=ml_dag,
    dag=dag,
)

trigger_ml_dag = TriggerDagRunOperator(
task_id='trigger_ml_dag',
trigger_dag_id='ml_dag',
dag=dag,
)

trigger_processing_dag >> ml_subdag >> trigger_ml_dag
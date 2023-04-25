from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os, spacy
from common_files.operators.ml_operator import FeatureExtractor, DataProcessorForML, DataSplitter, ModelTester, ModelTrainer
from airflow import DAG

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

ml_dag = DAG(
    'ml_analysis_pipeline',
    default_args=default_args,
    description='Predict if a customer is interested in finance',
    schedule_interval='@daily',
)

feature_extractor = FeatureExtractor('outputs/data/output_analysis_UserWillChristodoulou_DataScience_TestData.csv')
extract_ml_features_target_task = PythonOperator(
    task_id='extract_ml_features_target',
    python_callable=feature_extractor.extract_ml_features_target,
    bash_command='echo "extract_ml_features_target_task is Running"',
    dag=ml_dag,
)

data_processor_for_ml = DataProcessorForML(X=feature_extractor.X, y=feature_extractor.y)
data_processor_for_ml_task = PythonOperator(
    task_id='data_processor_for_ml',
    python_callable=data_processor_for_ml.process_data,
    bash_command='echo "data_processor_for_ml_task is Running"',
    dag=ml_dag,
)

split_data_task = DataSplitter(X=data_processor_for_ml.features_df, y=data_processor_for_ml.target_series)
split_train_test_task = PythonOperator(
    task_id='split_train_test',
    python_callable=split_data_task.split_train_test,
    op_kwargs={'test_size': 0.2},
    bash_command='echo "split_train_test_task is Running"',
    dag=ml_dag
)

model_trainer = ModelTrainer(X_train=split_data_task.X_train, y_train=split_data_task.y_train)
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=model_trainer.train,
    op_kwargs={},
    bash_command='echo "train_model_task is Running"',
    dag=ml_dag,
)

model_tester = ModelTester(model=model_trainer.model, X_test=split_data_task.X_test, y_test=split_data_task.y_test)
test_model_task = PythonOperator(
    task_id='test_model',
    python_callable=model_tester.test,
    op_kwargs={},
    bash_command='echo "test_model_task is Running"',
    dag=ml_dag,
)

# Set task dependencies
extract_ml_features_target_task >> split_train_test_task >> train_model_task >> test_model_task
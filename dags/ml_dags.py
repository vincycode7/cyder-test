from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
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

def get_attribute(task_instance, task_ids, attribute_name):
    """
    Dynamic function to retrieve the attribute of a previous DAG.
    """
    previous_task_instance = task_instance.xcom_pull(task_ids=task_ids)
    attribute_value = getattr(previous_task_instance, attribute_name, None)
    return attribute_value

ml_dag = DAG(
    'ml_analysis_pipeline',
    default_args=default_args,
    description='Predict if a customer is interested in finance',
    schedule_interval='@daily',
)

feature_extractor = FeatureExtractor('outputs/data/output_analysis_UserWillChristodoulou_DataScience_TestData.csv')
extract_ml_features_target_task = PythonOperator(
    task_id='feature_extractor',
    python_callable=feature_extractor.extract_ml_features_target,
    dag=ml_dag,
)

data_processor_for_ml = DataProcessorForML(data="{{ ti.xcom_pull(task_ids='feature_extractor', key='df') }}")
data_processor_for_ml_task = PythonOperator(
    task_id='data_processor_for_ml',
    python_callable=data_processor_for_ml.process_data,
    op_kwargs={'feature_extractor_attribute': get_attribute},
    dag=ml_dag,
)

split_train_test = DataSplitter(X="{{ ti.xcom_pull(task_ids='data_processor_for_ml', key='features_df') }}", y="{{ ti.xcom_pull(task_ids='data_processor_for_ml', key='target_series') }}")
split_train_test_task = PythonOperator(
    task_id='split_train_test',
    python_callable=split_train_test.split_train_test,
    op_kwargs={'test_size': 0.2, 'data_processor_for_ml_attribute': get_attribute},
    dag=ml_dag
)

model_trainer = ModelTrainer(X_train="{{ ti.xcom_pull(task_ids='split_train_test', key='X_train') }}", y_train="{{ ti.xcom_pull(task_ids='split_train_test', key='y_train') }}")
train_model_task = PythonOperator(
    task_id='model_trainer',
    python_callable=model_trainer.train,
    op_kwargs={'split_train_test_attribute': get_attribute},
    dag=ml_dag,
)

model_tester = ModelTester(model="{{ ti.xcom_pull(task_ids='model_trainer', key='model') }}", X_test="{{ ti.xcom_pull(task_ids='split_train_test', key='X_test') }}", y_test="{{ ti.xcom_pull(task_ids='split_train_test', key='y_test') }}")
test_model_task = PythonOperator(
    task_id='model_tester',
    python_callable=model_tester.test,
    op_kwargs={'model_trainer_attribute': get_attribute, 'split_train_test_attribute': get_attribute},
    dag=ml_dag,
)

# Set task dependencies
extract_ml_features_target_task >> split_train_test_task >> train_model_task >> test_model_task

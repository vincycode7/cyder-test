from typing import List, Tuple
import pandas as pd
# from airflow.operators.papermill_operator import PapermillOperator
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import os, spacy
from common_files.operators.data_operators import DataAnalyzer
from common_files.operators.ml_operator import FeatureExtractor, DataSplitter, ModelTester, ModelTrainer

in_notebook_mode = os.getenv('IN_NOTEBOOK_MODE', 'False').lower() == 'true'
print(f"in_notebook_mode: {in_notebook_mode}")

dag = DAG(
    'ml_analysis_pipeline',
    start_date=datetime(2023, 4, 25),
    schedule_interval=None
)

def load_or_download_models():
    spacy_model_name = 'en_core_web_sm'
    try:
        nlp = spacy.load(spacy_model_name)
    except OSError:
        spacy.cli.download(spacy_model_name)
        try:
            nlp = spacy.load(spacy_model_name)
        except Exception as e:
            raise Exception(f"Failed Process Due to {e}")
    output_dir="outputs/data/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  
    print(f"os.path.exists(output_dir): {os.path.exists(output_dir)}")
    
download_spacy_modls = PythonOperator(
    task_id='download_spacy_modls',
    python_callable=load_or_download_models,
    dag=dag
)

# Define the tasks
data_processor = DataAnalyzer(link_to_csv='inputs/UserWillChristodoulou_DataScience_TestData.csv')
analyze_data = PythonOperator(
    task_id='analyze_data',
    python_callable=data_processor.analyze_data,
    dag=dag
)

feature_extractor = FeatureExtractor(input_file='cyder-test/outputs/output_UserWillChristodoulou_DataScience_TestData.csv')
extract_ml_features_target = PythonOperator(
    task_id='extract_ml_features_target',
    python_callable=feature_extractor.extract_ml_features_target,
    dag=dag
)

data_splitter = DataSplitter(X=None, y=None)
split_train_test = PythonOperator(
    task_id='split_train_test',
    python_callable=data_splitter.split_train_test,
    op_kwargs={'test_size': 0.2},
    dag=dag
)

model_trainer = ModelTrainer(X_train=None, y_train=None)
train = PythonOperator(
    task_id='train',
    python_callable=model_trainer.train,
    dag=dag
)

model_tester = ModelTester(model=None, X_test=None, y_test=None)
test = PythonOperator(
    task_id='test',
    python_callable=model_tester.test,
    dag=dag
)


# Set task dependencies
download_spacy_modls >> analyze_data >> extract_ml_features_target >> split_train_test >> train >> test
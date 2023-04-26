from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from common_files.operators.ml_operator import FeatureExtractor, DataProcessorForML, DataSplitter, ModelTester, ModelTrainer
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.task_group import TaskGroup

def get_attribute(task_instance, task_ids, attribute_name):
    """
    Dynamic function to retrieve the attribute of a previous DAG.
    """
    previous_task_instance = task_instance.xcom_pull(task_ids=task_ids)
    attribute_value = getattr(previous_task_instance, attribute_name, None)
    return attribute_value

def ml_analysis_subdag(parent_dag_name, child_dag_name, default_args):
    with TaskGroup(group_id=f'{parent_dag_name}_{child_dag_name}') as subdag:

        feature_extractor = FeatureExtractor('outputs/data/output_analysis_UserWillChristodoulou_DataScience_TestData.csv')
        extract_ml_features_target_task = PythonOperator(
            task_id='feature_extractor',
            python_callable=feature_extractor.extract_ml_features_target,
        )

        data_processor_for_ml = DataProcessorForML(data="{{ ti.xcom_pull(task_ids='feature_extractor', key='df') }}")
        data_processor_for_ml_task = PythonOperator(
            task_id='data_processor_for_ml',
            python_callable=data_processor_for_ml.process_data,
            op_kwargs={'task_instance': '{{ ti }}', 'task_ids': extract_ml_features_target_task.task_id, 'attribute_name': 'df'},
        )

        with TaskGroup(group_id='ml_tasks') as ml_tasks:

            split_train_test = DataSplitter(X="{{ ti.xcom_pull(task_ids='data_processor_for_ml', key='features_df') }}", y="{{ ti.xcom_pull(task_ids='data_processor_for_ml', key='target_series') }}")
            split_train_test_task = PythonOperator(
                task_id='split_train_test',
                python_callable=split_train_test.split_train_test,
                op_kwargs={'test_size': 0.2, 'task_instance': '{{ ti }}', 'task_ids': 'data_processor_for_ml', 'attribute_name': 'features_df'},
            )

            model_trainer = ModelTrainer(X_train="{{ ti.xcom_pull(task_ids='split_train_test', key='X_train') }}", y_train="{{ ti.xcom_pull(task_ids='split_train_test', key='y_train') }}")
            train_model_task = PythonOperator(
                task_id='model_trainer',
                python_callable=model_trainer.train,
                op_kwargs={'task_instance': '{{ ti }}', 'task_ids': 'split_train_test', 'attribute_name': 'X_train'},
            )

            model_tester = ModelTester(model="{{ ti.xcom_pull(task_ids='model_trainer', key='model') }}", X_test="{{ ti.xcom_pull(task_ids='split_train_test', key='X_test') }}", y_test="{{ ti.xcom_pull(task_ids='split_train_test', key='y_test') }}")
            test_model_task = PythonOperator(
                task_id='model_tester',
                python_callable=model_tester.test,
                op_kwargs={'task_instance': '{{ ti }}', 'task_ids': 'split_train_test', 'attribute_name': 'y_test'},
            )
    

        end_task = DummyOperator(
            task_id='end'
        )

        # Set task dependencies
        extract_ml_features_target_task >> data_processor_for_ml_task >> ml_tasks >> end_task
        return subdag

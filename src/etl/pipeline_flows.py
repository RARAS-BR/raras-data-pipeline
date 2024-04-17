import logging
from prefect import flow
from prefect.blocks.system import String, Secret, JSON
from src.etl.extract_functions import extract_from_mongodb, extract_from_redcap
from src.etl.transform_functions import clean_data, preprocess_data
from src.etl.load_functions import load_forms_to_mongodb, load_project_to_mongodb


flow_run_template = '{project_name}_at_{time}'


@flow(name='redcap_to_mongodb',
      flow_run_name=flow_run_template,
      description='Extracts data from REDCap, preprocesses it and loads it to MongoDB.',
      log_prints=True,
      retries=3,
      retry_delay_seconds=60)
def preprocessing_pipeline(
        project_name: str,
        time: str,
        api_url: str,
        token_name: str,
        instructions_path: str,
        load_db_name: str,
        redcap_api_kwargs_block: str
):
    logging.info('Starting preprocessing pipeline from project %s at %s',
                 project_name, time)
    # Args
    url_block = String.load(api_url)
    token_block = Secret.load(token_name)
    instructions_path = instructions_path
    mongo_url: str = String.load('mongodb-url').value
    db_name: str = String.load(load_db_name).value
    api_kwargs = JSON.load(redcap_api_kwargs_block).value

    # Extract
    project = extract_from_redcap(api_url=url_block.value,
                                  api_token=token_block.get(),
                                  **api_kwargs)
    # Transform
    project = preprocess_data(project, instructions_path)
    # Load
    load_project_to_mongodb(project, mongo_url, db_name)


@flow(name='data_cleaning',
      flow_run_name=flow_run_template,
      description='Extracts preprocessed data from MongoDB, cleans it and loads it to MongoDB.',
      log_prints=True,
      retries=3,
      retry_delay_seconds=60)
def data_clean_pipeline(
        project_name: str,
        time: str,
        extract_db_name: str,
        instructions_file_path: str,
        load_db_name: str,
):
    logging.info('Starting preprocessing pipeline from project %s at %s',
                 project_name, time)
    # Args
    extract_db_name = String.load(extract_db_name).value
    instructions_file_path = instructions_file_path
    mongo_url = String.load('mongodb-url').value
    load_db_name = String.load(load_db_name).value

    # Extract
    forms, metadata = extract_from_mongodb(mongo_url, extract_db_name)
    # Transform
    collections = clean_data(forms, metadata, instructions_file_path)
    # Load
    load_forms_to_mongodb(collections, mongo_url, load_db_name)

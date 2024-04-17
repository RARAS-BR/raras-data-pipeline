from prefect import flow, task
from prefect.blocks.system import Secret, String
from src.redcap_project import REDCapProject

from src.etl.extract_functions import extract_from_redcap
from src.etl.transform_functions import preprocess_data
from src.etl.load_functions import load_project_to_mongodb


@task(name='extract_from_redcap', log_prints=True, retries=3, retry_delay_seconds=30)
def extract(api_url: str, api_token: str, **kwargs) -> REDCapProject:
    return extract_from_redcap(api_url, api_token, **kwargs)


@task(name='preprocessing', log_prints=True)
def transform(project: REDCapProject, instructions_path: str) -> REDCapProject:
    return preprocess_data(project, instructions_path)


@task(name='load_project_to_mongodb', log_prints=True)
def load(project: REDCapProject, mongo_url: str, db_name: str) -> None:
    return load_project_to_mongodb(project, mongo_url, db_name)


@flow(name='projeto_administrativo_preprocessing')
def run_flow():
    # Extract task args
    url_block = String.load('redcap-api-url')
    token_block = Secret.load('token-administrativo-joaobaiochi')

    # Transform task args
    data_cleaning_file_path = 'flows/admin/preprocessing/instructions.yaml'

    # Load task args
    mongo_url = 'mongodb://localhost:27017/'
    db_name = 'admin'

    project = extract(api_url=url_block.value,
                      api_token=token_block.get())
    project = transform(project, data_cleaning_file_path)
    load(project, mongo_url, db_name)


if __name__ == '__main__':
    run_flow()

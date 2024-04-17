from datetime import datetime
import pandas as pd
from prefect import flow
from src.etl.pipeline_flows import preprocessing_pipeline, data_clean_pipeline

pd.set_option('future.no_silent_downcasting', True)


@flow(name='prospectivo_2022_flow',
      log_prints=True)
def run_flow():
    current_time: str = datetime.now().strftime('%d-%m')
    project_name = 'prospectivo_2022'
    raw_layer_db_name = 'prospectivo22-raw-layer'
    cleaned_layer_db_name = 'prospectivo22-cleaned-layer'
    instructions_folder = 'flows/prospectivo22/instructions'
    redcap_api_kwargs = 'inquerito-kwargs'

    preprocessing_pipeline(
        project_name=project_name,
        time=current_time,
        api_url='redcap-api-url',
        token_name='token-prospectivo-2022',
        instructions_path=f'{instructions_folder}/preprocessing.yaml',
        load_db_name=raw_layer_db_name,
        redcap_api_kwargs_block=redcap_api_kwargs
    )

    data_clean_pipeline(
        project_name=project_name,
        time=current_time,
        extract_db_name=raw_layer_db_name,
        instructions_file_path=f'{instructions_folder}/data_cleaning.yaml',
        load_db_name=cleaned_layer_db_name,
    )


if __name__ == '__main__':
    run_flow()

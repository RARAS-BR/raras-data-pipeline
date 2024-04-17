from functools import reduce
from typing import List, Dict
import logging
import yaml
import pandas as pd
from prefect import task
from pandas import DataFrame
from src.preprocessing import Preprocessing
from src.data_cleaning import DataCleaning
from src.redcap_project import REDCapProject


@task(name='preprocessing', log_prints=True)
def preprocess_data(
        project: REDCapProject,
        instructions_path: str
) -> REDCapProject:
    """
    Preprocess a REDCap project data based on provided instructions.

    Parameters
    ----------
    project : REDCapProject
        The REDCap project to be cleaned.
    instructions_path : str
        The path to the YAML file containing the preprocessing instructions.

    Returns
    -------
    REDCapProject
        The cleaned REDCap project.
    """
    logging.info('Transform task: Data preprocessing.')

    with open(instructions_path, encoding='utf-8') as file:
        instructions = yaml.safe_load(file)

    pp = Preprocessing(project, instructions)
    project.update_forms(pp.forms)
    project.update_feature_map(pp.feature_map)
    project.update_outliers(pp.outliers)

    return project


@task(name='data_cleaning', log_prints=True)
def clean_data(
        forms: Dict[str, DataFrame],
        metadata: dict,
        instructions_path: str
) -> dict[str, DataFrame]:
    """
    Clean all forms in a REDCap project based on provided instructions.

    Parameters
    ----------
    forms : Dict[str, DataFrame]
        A dictionary where the keys are form names and the values are DataFrames representing the forms.
    metadata : Dict
        A dictionary containing codebook and other metadata from the project.
    instructions_path : str
        The path to a YAML file containing the data_cleaning instructions.

    Returns
    -------
    Dict[str, DataFrame]
        A dictionary where the keys are form names and the values are DataFrames representing the normalized forms.
    """
    logging.info('Transform task: %s', list(forms.keys()))

    with open(instructions_path, encoding='utf-8') as file:
        instructions = yaml.safe_load(file)

    data = {}
    for form_name, df in forms.items():
        if form_name not in instructions:
            continue
        logging.info('Transform task: Processing form "%s".', form_name)
        dc = DataCleaning(
            form_name=form_name,
            df=df,
            metadata=metadata,
            instructions=instructions[form_name]
        )
        dc.run_instructions()
        # TODO: temporary fix for NaT problem when loading into MongoDB
        dc.df = dc.df.replace({pd.NaT: None})

        data[form_name] = dc.df
        metadata['outliers'].update(dc.outliers)

    data['_metadata'] = metadata

    return data


# TODO: Find a better place to insert this post-processing function
def remove_records_without_required_forms(
        forms: Dict[str, DataFrame],
        required_forms: str | List[str],
        outliers: dict
) -> tuple[Dict[str, DataFrame], dict]:
    logging.info('Removing records without required forms.')
    logging.debug('Required forms: %s', required_forms)

    # Cast parameter to list
    if isinstance(required_forms, str):
        required_forms = [required_forms]

    # Get valid ids from required forms intersection
    required_records = [set(forms[col]['record_id']) for col in required_forms]
    valid_records = reduce(lambda a, b: a.intersection(b), required_records)

    # Drop records without required forms
    for form_name, df in forms.items():
        # Get invalid records
        invalid_records = set(df['record_id']) - valid_records
        # Drop records without required forms
        forms[form_name] = df[~df['record_id'].isin(invalid_records)]

        # Add values to outliers
        outliers['without_required_forms'] = {f'{form_name}': invalid_records}
        outliers['invalid_records'].update(invalid_records)

        if len(invalid_records) > 0:
            logging.info('remove_records_without_required_forms: removed %s records from %s.',
                         len(invalid_records), form_name)

    return forms, outliers

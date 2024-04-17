import logging
import pandas as pd
from prefect import task
from pandas import DataFrame
from pymongo import MongoClient
from src.redcap_project import REDCapProject


@task(name='extract_from_redcap', log_prints=True, retries=3, retry_delay_seconds=30)
def extract_from_redcap(
        api_url: str,
        api_token: str,
        **kwargs
) -> REDCapProject:
    """
    Extract data from a REDCap project using the API.

    Parameters
    ----------
    api_url : str
        The URL of the REDCap API.
    api_token : str
        The API token for accessing the REDCap project.
    **kwargs
        Additional keyword arguments to pass to the load_records method.

    Returns
    -------
    REDCapProject
        The loaded REDCap project.
    """
    logging.info('Extract task: Load data via REDCap API.')
    project = REDCapProject(api_url, api_token)
    project.load_records(**kwargs)

    return project


# Extract data from MongoDB and load each collection in a pandas DataFrame
@task(name='extract_from_mongodb', log_prints=True, retries=3, retry_delay_seconds=30)
def extract_from_mongodb(
        mongo_url: str,
        db_name: str,
        collections: str | list[str] = None
) -> tuple[dict[str, DataFrame], dict]:
    """
    Extract data from a MongoDB database and load each
    collection into a pandas DataFrame.

    Parameters
    ----------
    mongo_url : str
        The MongoDB connection string.
    db_name : str
        The name of the MongoDB database to extract data from.
    collections : str | list[str], optional
        The name or list of names of the collections to extract data from. If None,
        data will be extracted from all collections in the database. Defaults to None.

    Returns
    -------
    dict[str, DataFrame]
        A dictionary where the keys are the collection names and the values are
        pandas DataFrames containing the data from the corresponding collection.
    dict
        A dictionary containing metadata from the _metadata collection.
    """
    logging.info('Extract task: Load data via MongoDB.')

    client = MongoClient(mongo_url)
    db = client[db_name]

    if collections is None:
        collections = db.list_collection_names()
    elif isinstance(collections, str):
        collections = [collections]
    logging.info('Loading data frames: %s', collections)

    # Load each collection in a form
    forms = {}
    for collection in db.list_collection_names():
        forms[collection] = list(db[collection].find({}, {"_id": 0}))  # Exclude _id
        if collection != '_metadata':
            forms[collection] = pd.DataFrame(forms[collection])
    metadata: dict = forms['_metadata'][0]
    del forms['_metadata']

    client.close()

    return forms, metadata

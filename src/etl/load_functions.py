import logging
from prefect import task
from pymongo import MongoClient
from src.redcap_project import REDCapProject
from pandas import DataFrame


@task(name='load_project_to_mongodb', log_prints=True)
def load_project_to_mongodb(
        project: REDCapProject,
        mongo_url: str,
        db_name: str,
) -> None:
    """
    Load a REDCap project into a MongoDB database.

    Parameters
    ----------
    project : REDCapProject
        The REDCap project to load into MongoDB.
    mongo_url : str
        The MongoDB connection string.
    db_name : str
        The name of the MongoDB database to load the project into.

    Returns
    -------
    None
    """
    logging.info('Load task: Push to MongoDB.')

    client = MongoClient(mongo_url)
    db = client[db_name]

    # Save each form as a collection
    for form_name, df in project.forms.items():
        if df.empty:
            logging.warning("Form %s is empty.", form_name)
            continue
        collection = db[form_name]
        collection.delete_many({})
        collection.insert_many(df.to_dict(orient='records'))

    # Save metadata as a collection
    metadata = project.get_metadata()
    collection = db['_metadata']
    collection.delete_many({})
    collection.insert_one(metadata)

    client.close()


@task(name='load_forms_to_mongodb', log_prints=True)
def load_forms_to_mongodb(
        collections: dict,
        mongo_url: str,
        db_name: str,
) -> None:
    """
    Load processed data frames and codebook into MongoDB database.

    Parameters
    ----------
    collections : dict
        The processed data frames and codebook to load into MongoDB.
    mongo_url : str
        The MongoDB connection string.
    db_name : str
        The name of the MongoDB database to load the project into.

    Returns
    -------
    None
    """
    logging.info('Load task: Push to MongoDB.')

    client = MongoClient(mongo_url)
    db = client[db_name]

    # Save each form as a collection
    for form_name, df in collections.items():
        collection = db[form_name]
        collection.delete_many({})
        if isinstance(df, DataFrame):
            collection.insert_many(df.to_dict(orient='records'))
        elif isinstance(df, dict):
            collection.insert_one(df)
        else:
            raise ValueError(f'Invalid type {type(df)} for {form_name}')

    client.close()

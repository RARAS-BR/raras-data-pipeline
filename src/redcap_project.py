"""
This module contains the REDCapProject class which is used to interact with a REDCap project.
It includes methods for initializing the project, processing codebook, loading records, and making API calls.
"""

import logging
from io import StringIO
from typing import Optional

import requests
import numpy as np
import pandas as pd
from pandas import DataFrame
from src.handlers.api_handler import APIHandler
from src.handlers.metadata_handler import MetadataHandler


class REDCapProject:
    """
    A class to interact with a REDCap project.

    Parameters
    ----------
    api_url : str
        The URL for the REDCap API.
    api_token : str
        The API token for the REDCap project.
    missing_datacodes : dict, optional
        A dictionary of missing data codes (default is None).

    Attributes
    ----------
    project_id : str
        The project ID for the REDCap project.
    project_title : str
        The title of the REDCap project.
    df : DataFrame
        The records from the REDCap project.
    codebook : DataFrame
        The codebook from the REDCap project.
    identifier_fields : list
        A list of field names that are identifiers.
        The repeating forms from the REDCap project.
    raw_label_map : dict
        A mapping of raw labels to their corresponding labels.
    outliers : dict
        A dictionary of outlier records.
    feature_map : dict[str, list[str]]
        A mapping of field names to field labels.
    forms : dict
        A dictionary of forms.
    missing_datacodes : dict
        A dictionary of missing data codes.
    branching_logic_tree : dict
        A dictionary of branching logic.
    """

    def __init__(
            self,
            api_url: str,
            api_token: str,
            missing_datacodes: dict[str, str] = None
    ) -> None:
        # Instance handlers
        self.api = APIHandler(api_url, api_token)
        self.mh = MetadataHandler(self.api)

        # Initialize attributes
        self.project_id: Optional[str] = None
        self.project_title: Optional[str] = None
        self.df: pd.DataFrame() = None
        self.dag: pd.DataFrame() = None
        self.instruments: pd.DataFrame() = None
        self.repeating_forms_events: pd.DataFrame() = None
        self.codebook: pd.DataFrame() = None
        self.identifier_fields: list = []
        self.raw_label_map: dict = {}
        self.branching_logic_tree: dict = {}
        self.feature_map: dict[str, list[str]] = {}
        self.forms: dict[str, DataFrame] = {}
        self.outliers: dict = {
            'incomplete_records': [],
            'unverified_records': [],
            'duplicated_values': [],
            'invalid_records': {},
            'unmapped_labels': {},
            'missing_datacodes': {},
        }

        # Initialize project
        self.init_project()

        # Default missing data codes map
        if missing_datacodes is None:
            self.missing_datacodes = {
                'NI': 'No information',
                'INV': 'Invalid',
                'UNK': 'Unknown',
                'NASK': 'Not asked',
                'ASKU': 'Asked but unknown',
                'NAV': 'Temporarily unavailable',
                'MSK': 'Masked',
                'NA': 'Not applicable',
                'NAVU': 'Not available',
                'NP': 'Not present',
                'OTH': 'Other'
            }

    def init_project(self) -> None:
        """
        Sets the project information and codebook for the API connection.
        """
        # Get project information
        self.project_id, self.project_title = self.mh.load_project_info()
        # Load codebook
        self._instance_dag()
        self._instance_instruments()
        self._instance_repeating_forms_events()
        self._instance_codebook()
        self._instance_identifier_fields()
        self._instance_raw_label_map()

        logging.info('Project ID: %s', self.project_id)
        logging.info('Project Title: %s', self.project_title)
        logging.info('Data Access Groups (n): %s\n', self.dag.shape[0])

    def _instance_dag(self):
        self.dag = self.mh.load_metadata('dag')

    def _instance_instruments(self):
        self.instruments = self.mh.load_metadata('instrument')

    def _instance_repeating_forms_events(self):
        self.repeating_forms_events = self.mh.load_metadata('repeatingFormsEvents')

    def _instance_codebook(self):
        self.codebook = self.mh.load_metadata('metadata')

    def _instance_identifier_fields(self):
        _identifier_mask = self.codebook['identifier'] == 'y'
        self.identifier_fields = self.codebook[_identifier_mask]['field_name'].to_list()

    def _instance_raw_label_map(self):
        self.raw_label_map = self.mh.create_raw_label_map(self.codebook)

    def _instance_branching_logic_tree(self):
        self.branching_logic_tree = self.mh.create_branching_logic_tree(self.codebook)

    def update_forms(self, forms: dict[str, DataFrame]):
        self.forms = forms

    def update_feature_map(self, feature_map: dict[str, list[str]]):
        self.feature_map = feature_map

    def update_outliers(self, outliers: dict):
        for key, value in outliers.items():
            try:
                if isinstance(value, list):
                    self.outliers[key].extend(value)
                elif isinstance(value, dict):
                    self.outliers[key].update(value)
            except KeyError:
                self.outliers[key] = value

    def load_records(
            self,
            file_type: str = 'flat',
            file_format: str = 'csv',
            records: list[str] = None,
            fields: list[str] = None,
            forms: list[str] = None,
            raw_or_label: str = 'raw',
            raw_or_label_headers: str = 'raw',
            export_checkbox_label: bool = False,
            export_data_access_groups: bool = True,
            label_columns: list[str] = None,
            map_dags: bool = False
    ) -> None:
        """
        Exports records from Project as a Pandas dataframe.

        Parameters
        ----------
        file_type : str, optional
            The format of the exported records (default is 'flat').
                flat - output as one record per row [default]
                eav - output as one data point per row
                    Non-longitudinal: Will have the fields - record*, field_name, value
                    Longitudinal: Will have the fields - record*, field_name, value, redcap_event_name
        file_format : str, optional
            The format of the exported records (default is 'csv').
        records : list of str, optional
            An array of record names specifying specific records you wish to pull (by default, all records are pulled).
        fields : list of str, optional
            An array of field names specifying specific fields you wish to pull (by default, all fields are pulled).
        forms : list of str, optional
            An array of form_name names you wish to pull records for (by default, all records are pulled).
        raw_or_label : str, optional
            'raw' [default], 'label' - export the raw coded values or labels for the options of multiple choice fields.
        raw_or_label_headers : str, optional
            'raw' [default], 'label' - (for 'csv' format 'flat' type only) for the CSV headers,
            export the variable/field names (raw) or the field labels (label).
        export_checkbox_label : bool, optional
            True, False [default] - specifies the format of checkbox field values specifically when exporting the data
            as labels (i.e., when rawOrLabel=label) in flat format (i.e., when type=flat).
        export_data_access_groups : bool, optional
            True, False [default] - specifies whether to export the 'redcap_data_access_group' field when data access
            groups are utilized in the project.
        label_columns : list of str, optional
            A list of column names to load as labels for the data, common used when raw data have random values
            generated by a SQL query. Defaults to None.
        map_dags: bool, optional
            Whether to map redcap_data_access_group with data_access_group_id from redcap.dags. Defaults to False.

        Returns
        -------
        None

        Args:

        """
        # Load data
        response = self.api.make_api_call(
            'record', type=file_type, format=file_format, records=records,
            fields=fields, forms=forms, rawOrLabel=raw_or_label, rawOrLabelHeaders=raw_or_label_headers,
            exportCheckboxLabel=export_checkbox_label, exportDataAccessGroups=export_data_access_groups)
        if response.status_code != 200:
            raise requests.HTTPError(response.text)
        self.df = pd.read_csv(StringIO(response.text), low_memory=False)

        # Add custom labels to loaded data
        if label_columns:
            response = self.api.make_api_call(
                'record', type=file_type, format=file_format, records=records, forms=forms,
                fields=fields, rawOrLabel='label')
            if response.status_code != 200:
                raise requests.HTTPError(response.text)

            label_df = pd.read_csv(StringIO(response.text), low_memory=False)

            assert label_df.shape[0] == self.df.shape[0], \
                (f'Label data frame has different number of rows({label_df.shape[0]}) '
                 f'than raw data frame({self.df.shape[0]}).')

            # Check for unmapped values
            self.outliers['unmapped_labels'] = self.check_unmapped_labels(label_df, label_columns)
            # Replace raw values with labels
            self.df[label_columns] = label_df[label_columns]

        # Map redcap_data_access_group with data_access_group_id from redcap.dags
        if map_dags:
            logging.info('Mapping redcap_data_access_group with DAG name')
            replace_map: dict = self.dag.set_index('unique_group_name').to_dict()['data_access_group_name']
            self.df['redcap_data_access_group'] = self.df['redcap_data_access_group'].replace(replace_map)

    def get_metadata(self) -> dict[str, any]:
        """
        Generate metadata dictionary to load into NoSQL database.
        If data is dataframe, convert to dictionary.
        """
        return {
            'project_id': self.project_id,
            'project_title': self.project_title,
            'dag': self.dag.to_dict(orient='records'),
            'instruments': self.instruments.to_dict(orient='records'),
            'repeating_forms_events': self.repeating_forms_events.to_dict(orient='records'),
            'codebook': self.codebook.to_dict(orient='records'),
            'identifier_fields': self.identifier_fields,
            'raw_label_map': self.raw_label_map,
            'branching_logic_tree': self.branching_logic_tree,
            'feature_map': self.feature_map,
            'outliers': self.outliers,
        }

    def check_unmapped_labels(self, label_df, label_columns) -> list[dict] | None:
        all_nan_index = (
            label_df[label_columns].isna().all(axis=1)
            .loc[lambda x: x].index
        )
        unmapped_values = self.df.loc[all_nan_index, label_columns].stack().values
        if unmapped_values.any():
            logging.warning("The following values(raw) doesn't have a label associated: %s",
                            np.unique(unmapped_values))
            # Store unmapped values in outliers along with respective record_id
            unmapped_df = self.df.loc[all_nan_index, ['record_id'] + label_columns]
            unmapped_df = unmapped_df.melt(
                id_vars='record_id',
                value_vars=label_columns,
            ).dropna(subset=['value'])

            return unmapped_df.to_dict(orient='records')
        return None

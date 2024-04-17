import logging
import re
import json

import pandas as pd
import requests
from pandas import DataFrame
from src.handlers.api_handler import APIHandler


class MetadataHandler:
    def __init__(self, api: APIHandler):
        self.api = api

        self.project_id = None
        self.project_title = None
        self.codebook = None
        self.identifier_fields = None
        self.raw_label_map: dict = {}

    def load_metadata(self, content: str) -> DataFrame | None:
        """Load codebook from API."""
        try:
            response = self.api.make_api_call(content, format='json')

            if response.status_code == 200:
                return pd.DataFrame(response.json())
            else:
                logging.warning('Error: Received status code %s while fetching %s.',
                                response.status_code, content)
                return None
        except requests.exceptions.RequestException as e:
            logging.error('Error: An exception occurred while fetching %s: %s', content, str(e))

    def load_project_info(self) -> tuple[str, str]:
        """Load project info and project title from API."""
        response = self.api.make_api_call('project', format='json')
        data = response.json()

        return data['project_id'], data['project_title']

    def get_metadata(self) -> dict[str, any]:
        """Generate metadata dictionary to load into NoSQL database."""
        metadata = {
            'project_id': self.project_id,
            'project_title': self.project_title,
            'codebook': self.codebook,
            'identifier_fields': self.identifier_fields,
            'raw_label_map': self.raw_label_map
        }
        return metadata

    @staticmethod
    def create_raw_label_map(metadata: DataFrame) -> dict:
        # Transform column in mapping dict: key = raw, value = label
        raw_label = (
            metadata['select_choices_or_calculations']
            [metadata['field_type'].isin(['radio', 'checkbox', 'dropdown'])]
            # string example: "raw1, Label1 | raw2, Label2"
            .apply(lambda x: [re.split(', ', i, maxsplit=1)
                              for i in x.split(' | ') if ', ' in i])
            .apply(dict)
        )

        # Add field_name to index
        raw_label_map = pd.concat([
            metadata['field_name'],
            raw_label
        ], axis=1).dropna().set_index('field_name')

        # Cast to dictionary
        return raw_label_map.to_dict()['select_choices_or_calculations']

    @staticmethod
    def create_branching_logic_tree(codebook: DataFrame) -> dict:
        """Creates a branching logic tree from the codebook."""
        def make_map(list_child_parent):
            has_parent = set()
            all_items = {}
            for child, parent in list_child_parent:
                if parent not in all_items:
                    all_items[parent] = {}
                if child not in all_items:
                    all_items[child] = {}
                all_items[parent][child] = all_items[child]
                has_parent.add(child)

            result = {}
            for key, value in all_items.items():
                if key not in has_parent:
                    result[key] = value
            return result

        mask = codebook['branching_logic'] != ''
        mapping = codebook[mask].set_index('field_name').to_dict()['branching_logic']
        mapping = {other_column: re.search(r'\[(.*?)]', parent_column).group(1).split('(')[0]
                   for other_column, parent_column in mapping.items()}
        mapping = list(mapping.items())
        branching_logic_tree: dict = make_map(mapping)
        logging.debug(json.dumps(branching_logic_tree, indent=2))

        return branching_logic_tree

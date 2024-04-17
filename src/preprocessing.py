import copy
import re
import logging
from itertools import chain
from collections import Counter, defaultdict

import numpy as np
from pandas import DataFrame, Series

from src.redcap_project import REDCapProject
from src.handlers.transformer_handler import TransformerHandler


class Preprocessing:
    """
    The Preprocessing class is used to make transformations in
    a REDCap project without losing any original data.

    Attributes
    ----------
    df : DataFrame
        The DataFrame containing the data to be cleaned.
    codebook : DataFrame
        The DataFrame containing the metadata for the REDCap project.
    missing_datacodes_map : dict[str, str]
        A dictionary of missing data codes for the REDCap project.
    instructions : dict[str, any]
        A dictionary of instructions to be run on the data.
    feature_map : Series
        A Series containing the feature map for the REDCap project.
    forms : dict[str, DataFrame]
        A dictionary containing the forms for the REDCap project.
    outliers : dict
        A dictionary containing the outliers for the REDCap project.

    Parameters
    ----------
    redcap_project : REDCapProject
        The REDCapProject object containing the data to be cleaned.
    instructions : dict[str, any]
        A dictionary of instructions to be run on the data. The keys are the names
        of the methods to be run, and the values are dictionaries of parameters to
        be passed to the methods. If a method does not require
        parameters, the value should be None.
    """

    def __init__(
            self,
            redcap_project: REDCapProject,
            instructions: dict[str, any] = None
    ):
        # Initialize parameters
        self.df: DataFrame = redcap_project.df
        self.codebook: DataFrame = redcap_project.codebook
        self.missing_datacodes_map: dict = redcap_project.missing_datacodes
        self.instructions: dict = instructions

        self.feature_map: dict[str, list[str]] = {}
        self.forms: dict[str, DataFrame] = {}
        self.outliers = {}

        self.th = TransformerHandler()
        if instructions:
            self.th.run_instructions(self, self.instructions)

    def copy(self):
        """Return a deep copy of the Preprocessing instance."""
        return copy.deepcopy(self)

    def rename_instruments(self, mapping: dict[str, str]) -> None:
        """
        Renames instruments in the DataFrame and codebook.

        Parameters
        ----------
        mapping : dict[str, str]
            A dictionary of instruments names to be renamed. The keys are the
            original instruments names, and the values are the new instruments names.
        """
        logging.info('Renaming instruments.')
        logging.debug('Mapping: %s', mapping)

        # Rename repeat instruments
        self.df['redcap_repeat_instrument'] = self.df['redcap_repeat_instrument'].replace(mapping)

        # Rename instruments in codebook
        self.codebook['form_name'] = self.codebook['form_name'].replace(mapping)

        # Rename '_complete' columns
        rename_dict = {col: mapping[col.removesuffix('_complete')] + '_complete'
                       for col in self.df.filter(regex='_complete$').columns
                       if col.removesuffix('_complete') in mapping.keys()}
        self.df = self.df.rename(columns=rename_dict)

    def remove_missing_datacodes(self) -> None:
        """
        Removes missing data codes from the DataFrame. It first creates a
        dictionary to replace missing data codes with np.nan. Then, it finds missing
        data codes in the DataFrame and replaces them with np.nan.

        After that, it filters remaining missing data codes represented as features.
        The method then drops these columns from the DataFrame.
        """
        logging.info('Removing missing data codes.')

        # Get missing_datacodes represented as features(columns)
        _pattern = '|'.join(['__' + i.lower() for i in self.missing_datacodes_map.keys()])
        missing_datacodes_features = self.df.filter(regex=_pattern).columns

        # Extract missing datacodes values
        missing_values = (self.df
                          .map(lambda x: x if x in self.missing_datacodes_map.keys() else np.nan)
                          .dropna(how='all')
                          .dropna(axis=1, how='all')
                          )
        missing_values = {
            column: missing_values[column].value_counts().to_dict()
            for column in missing_values.columns
        }
        # Extract missing datacodes values in checkboxes
        missing_checkboxes = (self.df[missing_datacodes_features]
                              .replace({0: np.nan})
                              .dropna(how='all')
                              .dropna(axis=1, how='all'))
        missing_checkboxes = {column: missing_checkboxes[column].sum()
                              for column in missing_checkboxes.columns}

        missing_checkboxes_dict = defaultdict(dict)
        for key, value in missing_checkboxes.items():
            main_key, sub_key = key.split('___')
            missing_checkboxes_dict[main_key][sub_key.upper()] = value

        # Join values and store in outliers
        missing_values.update(missing_checkboxes_dict)
        self.outliers['missing_datacodes'] = missing_values

        # Create a dictionary to replace missing datacodes with np.nan
        replace_dict = {code: np.nan for code in self.missing_datacodes_map.keys()}

        # Find missing datacodes in df and replace with np.nan
        self.df = self.df.replace(replace_dict).infer_objects(copy=False)

        # Drop columns
        self.df.drop(columns=missing_datacodes_features, inplace=True)

    def aggregate_columns(self, agg_map: list[dict[str, str]]):
        """Helper function to apply aggregate_columns method."""
        for params in agg_map:
            self._apply_aggregate_columns(**params)

    def _apply_aggregate_columns(
            self,
            search_str: str,
            col_name: str = None,
            join_str: str = None
    ) -> None:
        """
        This method applies aggregate columns in the DataFrame.

        It first sets the default value for new_col_name by removing leading/trailing
        underscores from the search string. It then maps column names containing the
        search string and gets the first element (insert index position).

        The method subsets columns that start/end with [search_str] and groups them.
        If there are duplicate indexes, it groups them by a list/array.

        The method then drops the original columns, updates the column in position
        [insert_index], and updates the codebook(metadata).

        Parameters
        ----------
        search_str : str
            The string to search for in the column names.
        col_name : str, optional
            The name of the new column. Defaults to None.
        join_str : str, optional
            The string to join the grouped columns. Defaults to None.
        """
        # Default value for new_col_name (remove leading/trailing underscores)
        if col_name is None:
            col_name = re.sub(r'\b[_\W]+|[_\W]+\b', '', search_str)

        # Map column names containing the search string
        col_map = {index: col for index, col in enumerate(self.df.columns) if re.search(search_str, col)}
        assert len(col_map) > 0, f'No column contains {search_str}'
        logging.info('%s: columns matched in the search: %s', col_name, list(col_map.values()))

        # Get the first element (insert index position)
        insert_index = next(iter(col_map.keys()))

        # Subset columns that start/end with [search_str]
        sub_df = self.df[col_map.values()].copy()

        # Group columns
        sub_df = sub_df.stack().reset_index(level=1, drop=True)

        # Check for duplicate indexes, group by in a list if true
        if sub_df.index.duplicated(keep=False).any():
            logging.warning('Multiple values found in %s, aggregating by list/array.', search_str)
            sub_df = sub_df.groupby(level=0).agg(list)
            if join_str:
                sub_df = sub_df.apply(join_str.join)

        # Drop original columns
        self.df = self.df.drop(columns=col_map.values())

        # Update column in position [insert_index]
        # TODO: find why sometimes PerformanceWarning is raised
        self.df.insert(insert_index, col_name, sub_df)

        # Update codebook
        mapping = {key: col_name for key in col_map.values()}
        self.codebook.loc[:, 'field_name'] = self.codebook['field_name'].replace(mapping)
        self.codebook = self.codebook.drop_duplicates(subset='field_name')

    def decode_checkbox(self):
        """
        Decodes checkbox fields in the DataFrame. It first sets up masks
        to identify checkbox fields in the codebook. It then iterates over each
        checkbox field, setting choices and selecting columns to decode based on the
        field name.

        The method creates a copy of the columns to decode, removes the field name
        prefix from column names, and checks if the data has different values than
        boolean ones (0 or 1). If the data has different values, it raises an
        AssertionError.

        The method then replaces each value with the respective column name, sets
        empty list/values to NaN, drops the original encoded columns, and adds the
        decoded column to the DataFrame.

        Raises:
            ValueError: If a field is not found in the codebook.
            AssertionError: If the data has values different from 0, 1, or NaN.
        """
        # Setup masks
        checkbox_mask: Series = self.codebook['field_type'] == 'checkbox'
        fields: list[str] = self.codebook.loc[checkbox_mask, 'field_name'].tolist()

        for field_name in fields:

            assert field_name not in self.df.columns, f'Field {field_name} already exists in dataframe'

            field_mask: Series = self.codebook['field_name'] == field_name
            if sum(field_mask) == 0:
                raise ValueError(f'Field {field_name} not found in codebook.')

            # Set choices
            choices = self.codebook.loc[
                checkbox_mask & field_mask,
                'select_choices_or_calculations'].tolist()[0]

            # Subset raw values
            choices = [pair.split(',')[0].strip().lower().replace('-', '_')
                       for pair in choices.split('|')]

            # Select columns of prefix [field_name]
            decoded_columns = [field_name + '___' + choice for choice in choices]

            # Find index for the first occurence
            insert_index: int = self.df.columns.get_loc(decoded_columns[0])

            # Avoid selecting columns not loaded by REDCap
            cols_to_decode = list(set(decoded_columns).intersection(self.df.columns))
            cols_to_decode.sort()
            missing_cols = list(set(decoded_columns).difference(self.df.columns))
            logging.info('Decoding %s', cols_to_decode)
            if missing_cols:
                logging.warning('Missing columns in dataframe: %s', missing_cols)

            # Create a copy of the columns to decode
            aux_df: DataFrame = self.df[cols_to_decode].copy()

            # Remove field_name prefix from column names
            aux_df.columns = aux_df.columns.str.replace(field_name + '___', '')

            # Check if data has different values than boolean ones (0 or 1)
            assert aux_df.isin([0, 1, np.nan]).all().all(), 'Data has values different than 0, 1 or NaN'

            # Replace each value with the respective column name
            decoded_col: Series = aux_df.apply(lambda x: x.index[x == True].values, axis=1)  # Ignore PEP8 E712 (C0121)

            # Set empty list/values to NaN
            decoded_col = decoded_col.apply(lambda x: np.nan if len(x) == 0 else x)

            # Drop original encoded columns
            self.df = self.df.drop(columns=cols_to_decode)

            assert insert_index <= self.df.shape[0], \
                f'Index {insert_index} is greater than the number of columns in the dataframe'
            # Add decoded column to dataframe
            # TODO: find why sometimes PerformanceWarning is raised
            self.df.insert(insert_index, field_name, decoded_col)

    def _create_feature_map(self) -> None:
        """
        Creates a feature map for each form_name in the codebook.

        Raises:
            Warning: If there are duplicated forms in the feature map.
        """
        logging.info('Creating feature map.')

        # Get instruments that doesn't repeat
        single_instruments = set(self.codebook['form_name']) - set(self.df['redcap_repeat_instrument'].dropna())

        # Get the list of unique features for each form_name
        fields_mask = ~self.codebook['field_type'].isin(['descriptive'])
        feature_map: dict[str, list] = (
            self.codebook.loc[fields_mask, ['form_name', 'field_name']]
            .groupby('form_name')
            .agg(list)
            .to_dict()['field_name']
        )

        mapped_field_names = set(chain(*feature_map.values()))
        all_fields = set(self.codebook['field_name'])
        logging.info('Excluded fields(descriptive): %s', all_fields.difference(mapped_field_names))

        # Add identifier, instance_id and _complete fields
        for form_name in feature_map:
            if 'record_id' not in feature_map[form_name]:
                feature_map[form_name].insert(0, 'record_id')
            if 'redcap_data_access_group' not in feature_map[form_name]:
                feature_map[form_name].insert(1, 'redcap_data_access_group')
            if form_name not in single_instruments:
                feature_map[form_name].insert(2, 'redcap_repeat_instance')
            feature_map[form_name].extend([f'{form_name}_complete'])

        # Check for duplicated index - modify to accept dict instead of a pandas Series
        duplicated_keys = [item for item, count in Counter(feature_map.keys()).items() if count > 1]
        if duplicated_keys:
            logging.warning('Groupping duplicated forms: %s', duplicated_keys)
            # Grouping duplicated keys by summing their values
            for key in duplicated_keys:
                feature_map[key] = list(set(chain(*[value for item, value in feature_map.items() if item == key])))

        self.feature_map = feature_map

    def _cast_array_to_list(self):
        field_type_mask = self.codebook['field_type'] == 'checkbox'
        array_fields = self.codebook.loc[field_type_mask, 'field_name'].tolist()

        def convert_array_to_list(element):
            if isinstance(element, np.ndarray):
                return element.tolist()
            return element

        self.df[array_fields] = self.df[array_fields].apply(
            lambda x: x.apply(convert_array_to_list)
        )

    def subset_forms(self) -> None:
        """
        Subsets forms in the DataFrame. It first creates a feature map for
        each form_name in the codebook. Then, it iterates over each form_name in the
        feature map, subsetting data for repeating and single instruments.

        The method then checks if all forms in the feature map have a field_name and if
        any dataframes are empty.

        Raises:
            KeyError: If a feature map for a form_name is missing a field_name.
            Error: If there are empty forms.
        """
        logging.info('Subsetting forms.')

        # Create feature map
        self._create_feature_map()

        # Convert numpy arrays to lists to avoid errors
        # when loading to MongoDB
        self._cast_array_to_list()

        forms = {}
        for form_name, form_features in self.feature_map.items():
            # Subset data for repeating instruments
            if form_name in self.df['redcap_repeat_instrument'].unique():
                form_mask = self.df['redcap_repeat_instrument'] == form_name
                form_rows = self.df.loc[form_mask, form_features].copy()
            # Subset data for single instruments
            else:
                form_mask = self.df['redcap_repeat_instrument'].isna()
                form_rows = self.df.loc[form_mask, form_features].copy()
                # Remove empty data from other single instruments
                id_cols = ['record_id', 'redcap_data_access_group', form_features[-1]]  # "_complete" column
                subset_columns = form_rows.drop(columns=id_cols).columns
                form_rows = form_rows.dropna(how='all', subset=subset_columns)
            forms[form_name] = form_rows

        # Check if all forms in feature_map have a field_name
        for form_name in forms:  # Ignore lynt C0206, will lead to data frame reference errors.
            if not forms[form_name].columns.isin(self.feature_map[form_name]).all():
                raise KeyError(f"Feature map for form_name '{form_name}' is missing a field_name.")

        # Check if any dataframes are empty
        empty_forms = [form_name for form_name, data_frame in forms.items() if data_frame.empty]
        if empty_forms:
            logging.error('Empty forms: %s', empty_forms)

        self.forms = forms

    def create_new_form(
            self,
            new_form: str,
            source_form: str,
            common_columns: list,
            unique_columns: list
    ) -> None:
        """
        Creates a new form in the DataFrame.

        It first logs the details of the new form, including its name, the original
        form it's based on, and the common and unique columns. It then creates a copy
        of the source form with the specified common and unique columns and assigns
        it to the new form. Finally, it removes the unique columns from the original form.

        Parameters
        ----------
        new_form : str
            The name of the new form to be created.
        source_form : str
            The name of the original form the new form is based on.
        common_columns : list
            The list of common columns to be included in the new form.
        unique_columns : list
            The list of unique columns to be included in the new form.
        """
        logging.info('Creating new form.')
        logging.debug('New form: %s', new_form)
        logging.debug('Original form: %s', source_form)
        logging.debug('Common columns: %s', common_columns)
        logging.debug('Unique columns: %s', unique_columns)
        complete_column: str = f'{source_form}_complete'
        new_form_columns = common_columns + unique_columns + [complete_column]
        self.forms[new_form] = self.forms[source_form][new_form_columns].copy()
        self.forms[source_form] = self.forms[source_form].drop(columns=unique_columns)

    def merge_forms(
            self,
            target_form: str,
            source_form: str,
            merge_key: str | list[str] = None,
            drop_source: bool = True
    ) -> None:
        """
        This method merges two forms in the DataFrame.

        It first logs the details of the target and source forms. If no merge key is
        provided, it defaults to ['record_id', 'redcap_data_access_group']. It then
        merges the source form into the target form based on the merge key.

        If drop_source is True, it removes the source form from the DataFrame.

        Parameters
        ----------
        target_form : str
            The name of the target form.
        source_form : str
            The name of the source form to be merged into the target form.
        merge_key : str | list[str], optional
            The key(s) on which to merge the forms. Defaults to None.
        drop_source : bool, optional
            Whether to drop the source form after merging. Defaults to True.
        """
        logging.info('Merging forms.')
        logging.debug('Target form: %s', target_form)
        logging.debug('Source form: %s', source_form)

        # Default merge key
        if merge_key is None:
            merge_key = ['record_id', 'redcap_data_access_group']

        # Merge form_name
        self.forms[target_form] = self.forms[source_form].merge(
            self.forms[target_form],
            on=merge_key)

        if drop_source:
            del self.forms[source_form]

    def match_forms_to_schema(
            self,
            schema: dict[str, any]
    ) -> None:
        """
        Helper method to match forms into a given schema.

        Schema example:
        {
            'create_new_forms': [{
                'new_form': 'foo',
                'source_form': 'bar',
                'common_columns': ['spam'],
                'unique_columns': ['ham', 'eggs']
            }],
            'merge_forms': [{
                'target_form': 'foo',
                'source_form': 'baz',
            }]
        }
        """
        logging.info('Matching forms to schema.')

        operations = {
            'create_new_forms': self.create_new_form,
            'merge_forms': self.merge_forms
        }

        for operation, method in operations.items():
            if operation in schema:
                for mapping in schema[operation]:
                    method(**mapping)

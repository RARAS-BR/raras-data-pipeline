import logging
import re
from functools import partial
from typing import Literal, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Index


class DataCleaning:
    """
    Class to process a REDCap form.

    Parameters
    ----------
    form_name : str
        The name of the form to be processed.
    df : DataFrame
        The DataFrame containing the form data.
    metadata : dict
        A dictionary containing the form metadata.
    instructions : dict[str, any], optional
        A dictionary containing the instructions to be executed.

    Attributes
    ----------
    form_name : str
        The name of the form to be processed.
    df : DataFrame
        The DataFrame containing the form data.
    codebook : DataFrame
        The DataFrame containing the form codebook.
    outliers : dict
        A dictionary containing the outliers for each form.
    raw_label_map : dict
        A dictionary containing the raw label mapping for each form.
    instructions : dict
        A dictionary containing the instructions to be executed.
    other_columns_map : dict
        A dictionary containing the mapping of "other" columns with the parent columns.
    """

    def __init__(
            self,
            form_name: str,
            df: DataFrame,
            metadata: dict,
            instructions: dict[str, any] = None
    ):
        self.form_name = form_name
        self.df = df
        self.codebook = pd.DataFrame(metadata['codebook'])
        self.raw_label_map = metadata['raw_label_map']
        self.instructions = instructions

        self.outliers = metadata['outliers']
        self.other_columns_map = {}

    def run_instructions(self):
        """Run all instructions."""
        for method_name, params in self.instructions.items():
            method = getattr(self, method_name, None)

            if method is None or not callable(method):
                raise ValueError(f"{method_name} is not a valid method of the FormProcessor class")

            if params is None:
                method()
            elif isinstance(params, list):
                method(*params)
            elif isinstance(params, dict):
                method(**params)
            else:
                raise ValueError(f"Invalid parameters for method {method_name}")

    def remove_duplicates(
            self,
            column: str,
            keep: Literal["first", "last", False] = 'last',
            drop: bool = True
    ) -> None:
        """
        This method removes duplicate records from the DataFrame based on a specified column.

        It first standardizes the specified column by converting it to lowercase, removing leading and trailing spaces,
        and removing special characters. It then identifies duplicate values within the same 'redcap_data_access_group'.

        If duplicates are found, they are added to the 'outliers' dictionary. If the 'keep' parameter is set, the method
        removes the last occurrence of duplicated values.

        Args:
            column (str): The name of the column to check for duplicates.
            keep (Literal["first", "last", False], optional): Determines which duplicates (if any) to keep.
                - 'first' : Keep the first occurrence.
                - 'last' : Keep the last occurrence.
                - False : Drop all duplicates.
                Defaults to 'last'.
            drop (bool, optional): Whether to remove the duplicates from the DataFrame. Defaults to True.

        Raises:
            Warning: If there are duplicated values in the specified column for the same 'redcap_data_access_group'.
            Debug: If duplicates are removed from the DataFrame.
        """
        logging.info('Removing duplicates in feature "%s".', column)

        # Standardize and remove special characters
        norm_col = f'{column}_std'
        self.df[norm_col] = self.df[column].str.lower().str.strip().str.replace('[^A-Za-z0-9 ]+', '', regex=True)

        # Get duplicated values for the same "redcap_data_access_group"
        duplicated_values: dict[str, list] = (
            self.df[self.df.duplicated(subset=norm_col, keep=False)]
            # Avoid capturing NaNs
            .dropna(subset=norm_col)
            .groupby(norm_col, group_keys=True).agg({
                'record_id': list,
                'redcap_data_access_group': 'nunique',
            })
            # Filter for duplicated values in the same group
            .loc[lambda x: x['redcap_data_access_group'] == 1, 'record_id']
            .to_dict()
        )
        del self.df[norm_col]

        if len(duplicated_values) > 0:
            # Add to outliers dict
            self.outliers['duplicated_values']: dict[str, dict] = {column: duplicated_values}
            logging.warning(
                'Duplicated values in feature "%s" for the same redcap_data_access_group: %s',
                column, len(duplicated_values)
            )

        if drop:
            # Remove duplicated values
            mask = self.df[column].isin(duplicated_values) & self.df.duplicated(subset=column, keep=keep)
            invalid_records: set = set(self.df[mask]['record_id'])

            # Remove invalid record_ids
            self.df = self.df[~self.df['record_id'].isin(invalid_records)]

            # Add to outliers dict
            self.outliers['invalid_records'].extend(invalid_records)
            logging.debug('remove_duplicates: removed %s records.', len(invalid_records))

    def remove_incomplete_forms(self) -> None:
        """
        This method removes incomplete forms from the DataFrame.

        It iterates over each form_name in the forms dictionary, selects the columns ending with '_complete',
        and creates a boolean mask to identify rows with incomplete/unverified forms. It then filters these rows
        out of the DataFrame.

        If there is more than one column ending with '_complete', it logs a warning.

        After filtering the DataFrame, it drops the '_complete' column and adds the number of incomplete and unverified
        records to the 'outliers' attribute of the instance.

        Raises:
            Warning: If there is more than one column ending with '_complete' in the DataFrame.
        """
        logging.info('Removing incomplete forms.')

        # Select column ending with '_complete'
        complete_column: Index = self.df.filter(regex='_complete$').columns

        if len(complete_column) == 0:
            logging.error('No column ending with "_complete" found in the DataFrame %s.',
                          self.form_name)

        if len(complete_column) > 1:
            logging.warning('More than one column ending with "_complete" in %s (%s). '
                            'This behavior can lead to unwanted filtered records,'
                            'the last column will be selected.',
                            self.form_name, complete_column)
            complete_column: str = complete_column.tolist()[-1]
        else:
            complete_column: str = complete_column[0]

        # Create a boolean mask to identify rows with incomplete/unverified forms
        incomplete_records_mask = self.df[complete_column].isin([0])
        unverified_records_mask = self.df[complete_column].isin([1])

        # Add values to outliers
        self.outliers['incomplete_records'].extend(
            {f'{self.form_name}': self.df[incomplete_records_mask]['record_id'].tolist()})
        self.outliers['unverified_records'].extend(
            {f'{self.form_name}': self.df[unverified_records_mask]['record_id'].tolist()})

        # Filter values
        self.df = self.df[~(incomplete_records_mask + unverified_records_mask)]

        # Drop _complete column
        self.df = self.df.drop(columns=complete_column)

    def replace_other_columns(
            self,
            search_str: str = None,
            column_map: dict[str, str] = None,
            drop_columns: bool = False
    ) -> None:
        """
        Replaces the values in the specified columns or in the columns defined in the other_columns_map attribute
        with the corresponding values from the other_columns_map.

        Parameters
        ----------
        search_str : str, optional
            The search string used to find the "other" columns in the codebook.
            If not provided, the method will use the search string from the other_columns_search_str attribute.
        column_map : dict[str, str], optional
            The list of column names to replace. If not provided, the method will use the keys from the
            other_columns_map attribute.
        drop_columns : bool, optional
            Whether to drop the other columns after replacing the values. Defaults to False.

        Raises
        ------
        ValueError
            If the create_other_columns_mapping method has not been called before this method.
            If no other columns were found in the codebook to replace.

        Notes
        -----
        This method normalizes the other column values to lowercase and gets the other value from the raw_label_map.
        If the other value does not contain the other_columns_search_str, the column is skipped.
        If there are arrays in the column values, the replace_array_item method is used to replace the array items.
        In the general case, the values in the column are replaced with the values from the other column if they
        match the other value.
        """
        if search_str:
            self._create_other_columns_mapping(search_str)

        mapping: dict = column_map or self.other_columns_map
        if not mapping:
            raise ValueError('No columns to replace.')

        for column, other_column in mapping.items():

            # Normalize other column values
            self.df[other_column] = self.df[other_column].apply(lambda x: x.lower() if isinstance(x, str) else x)
            # Get other value from raw_label_map
            other_value = list(self.raw_label_map[column].keys())[-1].lower()

            if search_str not in other_value:
                logging.warning("Skipping column %s since it doesn't contain '%s' value.",
                                other_column, other_value)

            # Case when there are arrays in the column values
            if (self.df[column].dropna().apply(isinstance, args=((list, np.ndarray),))).any():
                self.df[column] = [self._replace_array_item(index, value, self.df, other_column, other_value)
                                   for index, value in self.df[column].items()]
            # General case
            else:
                self.df[column] = np.where(
                    self.df[column] == other_value,  # boolean array to match
                    self.df[other_column],  # value if True (replace)
                    self.df[column])  # value if False (default)

        if drop_columns:
            self.df = self.df.drop(columns=list(mapping.values()))

    def remap_boolean_labels(self, columns: list[str], bool_map: dict = None):
        """
        Remaps the boolean labels in the specified columns of the DataFrame.

        This method replaces the boolean labels in the DataFrame with their respective integer values.
        The mapping of labels to integers is defined in the bool_map attribute.

        Parameters
        ----------
        columns : list[str]
            The list of column names to remap. If a single string is provided, it is converted into a list.
        bool_map : dict, optional
            The dictionary mapping boolean labels to integers. If not provided, the method will use the default mapping.

        Raises
        ------
        ValueError
            If the dtype of the column is not 'object'.
        """

        if bool_map is None:
            bool_map = {
                '0': 0,
                'não': 0,
                'nao': 0,
                '1': 1,
                'sim': 1,
                'nao_se_aplica': np.nan,
                'não se aplica': np.nan,
            }

        if isinstance(columns, str):
            columns = [columns]

        for col in columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                raise ValueError(f'Column {col} is not of object type, it is already a numeric type.')

            non_empty_values = self.df[col].dropna().count()

            self.df[col] = (self.df[col]
                            .dropna()
                            .str.lower()
                            .map(bool_map)
                            .infer_objects(copy=False)
                            .astype('Int64')
                            )

            new_non_empty_values = self.df[col].dropna().count()
            if non_empty_values != new_non_empty_values:
                logging.warning('Column %s: %s values were lost in the remapping.',
                                col, non_empty_values - new_non_empty_values)

    def remap_categorical_labels(self, columns: list[str] = None):
        """
        Remaps the categorical labels in the specified columns of the DataFrame.

        This method replaces the categorical labels in the DataFrame with their respective integer values.
        The mapping of labels to integers is defined in the raw_label_map attribute.

        Parameters
        ----------
        columns : list[str], optional
            The list of column names to remap. If not provided, the method will use the keys from the
            raw_label_map attribute.
        """
        columns = columns or self.raw_label_map.keys()
        for col in columns:
            if col in self.df.columns:
                mapping: dict = self.raw_label_map.get(col, {})
                # Convert categorical column to object if it's a numeric type
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].dropna().astype(int).astype(str)
                self.df[col] = self.df[col].apply(partial(self.replace_label, mapping=mapping))

    def normalize_string(
            self, columns: list[str],
            method: Literal['lower', 'upper', 'title', 'capitalize']
    ):
        """
        Normalizes the string values in the specified columns of the DataFrame.

        This method applies a string data_cleaning method to the values in the specified columns.
        The data_cleaning method can be one of the following: 'lower', 'upper', 'title', 'capitalize'.

        Parameters
        ----------
        columns : list[str]
            The list of column names to normalize. If a single string is provided, it is converted into a list.
        method : Literal['lower', 'upper', 'title', 'capitalize']
            The string data_cleaning method to apply.
            Must be one of the following: 'lower', 'upper', 'title', 'capitalize'.

        Raises
        ------
        ValueError
            If the specified method is not one of the allowed methods.
        """
        if isinstance(columns, str):
            columns = [columns]

        for col in columns:
            method_map = {
                'lower': self.df[col].str.lower,
                'upper': self.df[col].str.upper,
                'title': self.df[col].str.title,
                'capitalize': self.df[col].str.capitalize
            }

            if method not in method_map:
                raise ValueError(f'Invalid method {method}')

            self.df[col] = method_map[method]()

    def enforce_dtype(self, dtype_map: dict[str, list[str]]):
        """
        Enforces the specified data types on the columns of the DataFrame.

        This method changes the data type of the specified columns in the DataFrame to the specified data types.
        The mapping of columns to data types is defined in the dtype_map parameter.

        Parameters
        ----------
        dtype_map : dict[str, list[str]]
            The dictionary mapping data types to lists of column names. The keys are the data types and the values are
            lists of column names. The data types must be one of the following: 'Int64', 'Float64', 'datetime'.

        Raises
        ------
        ValueError
            If the keys of dtype_map are not one of the allowed data types.
        """

        if not set(dtype_map.keys()).issubset(['Int64', 'Float64', 'datetime']):
            raise ValueError('Invalid dtype_map keys')

        for dtype, columns in dtype_map.items():
            for col in columns:
                if dtype == 'datetime':
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                else:
                    try:
                        self.df[col] = self.df[col].astype(dtype, errors='raise')
                    except ValueError:
                        logging.error('Error when trying to convert column %s to %s. Attempting to fix.',
                                      col, dtype)
                        # Check numeric digits and get ouliters
                        valid_values, invalid_records = self._check_numeric(col)
                        self.df[col] = valid_values.astype(dtype, errors='raise')
                        # Store outliers
                        self.outliers['invalid_records'][col] = invalid_records

    def rename_features(self, mapping: dict[str, str]):
        """
        Renames the columns of the DataFrame.

        This method renames the columns of the DataFrame according to the specified mapping.
        The mapping is a dictionary where the keys are the current column names and the values are the new column names.

        Parameters
        ----------
        mapping : dict[str, str]
            The dictionary mapping current column names to new column names.

        Raises
        ------
        KeyError
            If a column specified in the mapping does not exist in the DataFrame.
        """
        try:
            self.df = self.df.rename(columns=mapping)
        except KeyError as ke:
            logging.error('KeyError: %s', ke)
            raise ke

    def drop_features(self, columns: str | list[str]):
        """
        Drops the specified columns from the DataFrame.

        This method removes the specified columns from the DataFrame. If a single string is provided, it is
        converted into a list.

        Parameters
        ----------
        columns : str | list[str]
            The list of column names to drop. If a single string is provided, it is converted into a list.

        Raises
        ------
        KeyError
            If a column specified in the list does not exist in the DataFrame.
        """
        if isinstance(columns, str):
            columns = [columns]
        try:
            self.df = self.df.drop(columns=columns)
        except KeyError as ke:
            logging.error('KeyError: %s', ke)
            raise ke

    def _create_other_columns_mapping(self, search_str: str) -> None:
        """
        Create a dictionary mapping "other" column with the parent column.

        Parameters
        ----------
        search_str : str
            The search string used to find the "other" columns in the codebook.

        Returns
        -------
        None
            The dictionary is stored in the other_columns_map attribute.
        """

        # Create dictionary mapping "other" column with the parent column
        form_mask = self.codebook['form_name'] == self.form_name
        field_type_mask = self.codebook['field_type'].isin(['radio', 'checkbox', 'text'])
        branching_logic_mask = self.codebook['branching_logic'] != ''
        field_name_mask = self.codebook['field_name'].str.lower().str.contains(search_str.lower())
        # Apply filters
        mapping = self.codebook[form_mask & field_type_mask & branching_logic_mask & field_name_mask]
        # Transform into dictionary
        mapping = mapping[['field_name', 'branching_logic']].set_index('field_name').to_dict()['branching_logic']
        # Extract the parent column name from branching logic field
        mapping = {re.search(r'\[(.*?)]', parent_column).group(1).split('(')[0]: other_column
                   for other_column, parent_column in mapping.items()}
        logging.info('Fill mapping:\n%s', mapping)

        if not mapping:
            raise ValueError('No other columns found in the codebook to replace with %s.', search_str)

        self.other_columns_map = mapping

    def _check_numeric(self, column):
        is_numeric = self.df[column].dropna().str.isnumeric()

        # Use the mask to get the valid and invalid values
        valid_values = self.df[column].dropna()[is_numeric]
        invalid_indexes = self.df[column].dropna()[~is_numeric].index
        invalid_records = self.df.loc[invalid_indexes, 'record_id'].tolist()

        if invalid_records.any():
            logging.warning('Invalid records found in %s: %s', column, invalid_records)

        return valid_values, invalid_records

    @staticmethod
    def replace_label(x, mapping: dict):
        """Replace values in a Series according to a mapping dictionary."""
        try:
            # Case when the column has been transformed by the decode_checkbox method
            if isinstance(x, (np.ndarray, list)):
                return [mapping[i] for i in x]
            # General case
            return mapping[x]
        except KeyError:
            return x

    @staticmethod
    def _replace_array_item(index: int, value: any, df: DataFrame, other_column: str, other_value: str) -> any:
        """Replace an item in an array if it contains the other_value."""
        if isinstance(value, (list, np.ndarray)):
            # Replace array item only if it contains the other_value
            return [df.loc[index, other_column] if item.lower() == other_value else item for item in value]

        return value

    @staticmethod
    def _extract_ids_and_descriptions(x: str, pattern: str) -> Union[list[str], float]:
        """ Extract IDs and descriptions from a string based on a pattern."""
        # Remove multiple whitespaces before applying the pattern
        x = re.sub(r'\s+', ' ', x)
        # Find all matches of the pattern in the string
        matches = re.findall(pattern, x)
        # If matches are found, join them and extract the description
        if matches:
            extracted_id = ' - '.join(matches)
            # Find the last match in the string
            last_match = matches[-1]
            # Find the position of the last match
            last_match_position = x.rfind(last_match)
            # Extract the description based on the position of the last match
            extracted_description = x[last_match_position + len(last_match) + 3:].strip()
        else:
            logging.warning('No matches found for %s.', x)
            return np.nan

        return [extracted_id, extracted_description]

    # TODO: Temporary custom functions built-in to the class
    def _scale_values(self, column: str, size: int, multiplier: int, period=True):
        size_mask = self.df[column].str.len() == size
        if period:
            dot_mask = self.df[column].str.contains('\\.', regex=True)
        else:
            dot_mask = True
        fixed_values = (self.df[column][size_mask & dot_mask]
                        .str.replace('.', '')
                        .astype('Int64')
                        .mul(multiplier))
        if fixed_values.any():
            self.df.loc[fixed_values.index, column] = fixed_values
            # Store outliers
            self.outliers['invalid_records'][column] = self.df.loc[fixed_values.index, 'record_id'].tolist()

    def scale_birth_weight(self):
        self._scale_values('peso_nascimento', 4, 10)
        self._scale_values('peso_nascimento', 3, 100)
        self._scale_values('peso_nascimento', 2, 1000)
        self._scale_values('peso_nascimento', 1, 1000, period=False)
        # Fix remaining valid values
        self.df['peso_nascimento'] = (self.df['peso_nascimento']
                                      .dropna()
                                      .str.replace('.', '')
                                      .astype('Int64'))

    def recalculate_age(
            self,
            age_column: str,
            age_unit_column: str,
            new_column: str,
            age_unit_map=None
    ):
        if age_unit_map is None:
            age_unit_map = {'year': 365, 'month': 30, 'day': 1}

        recalculated_values = (
                self.df[age_column].astype('Int64')
                * self.df[age_unit_column].replace(age_unit_map).infer_objects(copy=False)
        ).astype('Int64')

        # Insert  new values
        column_index = self.df.columns.get_loc(age_column)
        self.df.insert(column_index + 1, new_column, recalculated_values)

        self.df = self.df.drop(columns=[age_column, age_unit_column])

    def split_code_descrpition(self, *args):
        for kwargs in args:
            self.apply_split_code_descrpition(**kwargs)

    def apply_split_code_descrpition(
            self,
            column: str,
            filter_pattern: str,
            code_name: str,
            desc_name: str
    ):
        # Extract code and description
        self.df[column] = self.df[column].dropna().apply(
            lambda x: self._extract_ids_and_descriptions(x, filter_pattern))
        code = self.df[column].dropna().apply(lambda x: x[0].strip())
        description = self.df[column].dropna().apply(lambda x: x[1].strip())

        # Insert values
        column_index = self.df.columns.get_loc(column)
        self.df.insert(column_index + 1, f'{code_name}', code)
        self.df.insert(column_index + 2, f'{desc_name}', description)
        # Try to cast code column to int
        try:
            self.df[f'{code_name}'] = self.df[f'{code_name}'].astype(float).astype('Int64')
        except ValueError:
            logging.warning('Could not cast %s to int.', code_name)

        # Drop original column
        self.df = self.df.drop(columns=column)

    def check_cpf(self):
        invalid_cpf_mask = self.df['cpf'].dropna().apply(self._validate_cpf).isna()
        if invalid_cpf_mask.any():
            logging.warning('Invalid CPF found: %s', invalid_cpf_mask.sum())
            self.df['cpf'] = self.df['cpf'].mask(invalid_cpf_mask, np.nan)
            # Add to outliers
            invalid_records = self.df['record_id'].mask(~invalid_cpf_mask, np.nan).dropna().tolist()
            self.outliers['invalid_records']['cpf'] = invalid_records

    def check_cns(self):
        invalid_cns_mask = self.df['cns'].dropna().apply(self._validate_cns).isna()
        if invalid_cns_mask.any():
            logging.warning('Invalid CNS found: %s', invalid_cns_mask.sum())
            self.df['cns'] = self.df['cns'].mask(invalid_cns_mask, np.nan)
            # Add to outliers
            invalid_records = self.df['record_id'].mask(~invalid_cns_mask, np.nan).dropna().tolist()
            self.outliers['invalid_records']['cns'] = invalid_records

    def create_identifier_column(self):
        """Fix for Retrospectivo project."""
        identificador_idx = self.df.columns.get_loc('identificador')
        cpf_mask = self.df['identificador'][self.df['tipo_identificador'] == 'cpf']
        cns_mask = self.df['identificador'][self.df['tipo_identificador'] == 'cns']
        # Add CPF/CNS columns
        self.df.insert(identificador_idx, 'cpf', cpf_mask)
        self.df.insert(identificador_idx + 1, 'cns', cns_mask)
        # Drop unused columns
        self.df = self.df.drop(columns=['tipo_identificador', 'identificador'])

    @staticmethod
    def _validate_cpf(cpf: str, return_value: bool = True):
        false = np.nan if return_value else False

        # Cast to list and remove non-digit characters
        cpf = [int(i) for i in str(cpf) if i.isdigit()]

        # Check if the CPF has the correct number of digits
        if len(cpf) != 11:
            return false

        # Calculate the first verification digit
        product_sum = sum(a * b for a, b in zip(cpf[:9], range(10, 1, -1)))
        expected_digit = (product_sum * 10 % 11) % 10
        if cpf[9] != expected_digit:
            return false

        # Calculate the second verification digit
        product_sum = sum(a * b for a, b in zip(cpf[:10], range(11, 1, -1)))
        expected_digit = (product_sum * 10 % 11) % 10
        if cpf[10] != expected_digit:
            return false

        # Valid CPF
        return ''.join(str(i) for i in cpf) if return_value else True

    @staticmethod
    def _validate_cns(cns: str, return_value: bool = True):
        """
        https://integracao.esusab.ufsc.br/v211/docs/algoritmo_CNS.html
        """
        false = np.nan if return_value else False
        true = cns if return_value else True

        # Check size
        if len(cns.strip()) != 15:
            return false

        # Routine for numbers starting in 1 or 2
        if cns[0] in ['1', '2']:

            pis = cns[:11]
            soma = sum((int(pis[i]) * (15 - i)) for i in range(11))
            resto = soma % 11
            dv = 11 - resto if resto != 0 else 0

            if dv == 10:
                soma = sum((int(pis[i]) * (15 - i)) for i in range(11)) + 2
                resto = soma % 11
                dv = 11 - resto if resto != 0 else 0
                resultado = pis + "001" + str(int(dv))
            else:
                resultado = pis + "000" + str(int(dv))

            if cns == resultado:
                return true
            return false

        # Routine for numbers starting in 7,8 or 9
        if cns[0] in ['7', '8', '9']:

            soma = sum((int(cns[i]) * (15 - i)) for i in range(15))
            resto = soma % 11

            if resto == 0:
                return true
            return false
        return false

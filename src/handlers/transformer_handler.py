import logging
import numpy as np
from pandas import DataFrame, Series


class TransformerHandler:

    @staticmethod
    def run_instructions(transformer: any, instructions: dict[str, any]) -> None:
        """
        Run instructions for a transformer object.

        Parameters
        ----------
        transformer : any
            The transformer object to be used.
        instructions : dict[str, any]
            A dictionary containing the instructions to be executed.
            The keys are the method names and the values are the parameters to be passed to the method.

        Returns
        -------
        None
        """
        for method_name, params in instructions.items():
            method = getattr(transformer, method_name, None)

            if method is None or not callable(method):
                class_name = transformer.__class__.__name__
                raise ValueError(f"{method_name} is not a valid method of the {class_name} class")

            if params is None:
                method()
            else:
                method(**params)

    @staticmethod
    def decode_checkbox(
            df: DataFrame,
            codebook: DataFrame,
            form_name: str,
            field_name: str,
            join_str: str = None
    ) -> DataFrame:
        """
        Decodes a checkbox field in the form.

        This method replaces the encoded checkbox values in the DataFrame with their respective column names.
        The decoded values are then inserted back into the DataFrame as a new column.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to be modified.
        codebook : DataFrame
            The codebook containing the information about the fields(REDCap metadata).
        form_name : str
            The name of the form to be filtered in the codebook.
        field_name : str
            The name of the field to be decoded.
        join_str : str, optional
            String used to join the values if the resulting value is a list.

        Raises
        ------
        ValueError
            If the form or the field is not found in the codebook.
        AssertionError
            If the data has values different from 0, 1 or NaN.

        Notes
        -----
        This method creates a copy of the columns to decode, removes the field_name prefix from column names,
        checks if data has different values than boolean ones (0 or 1), replaces each value with the respective
        column name, joins values with join_str if provided, sets empty list/values to NaN, drops original encoded
        columns, and adds the decoded column to the DataFrame.
        """
        # Setup masks
        checkbox_mask: Series = codebook['field_type'] == 'checkbox'
        form_mask: Series = codebook['form_name'] == form_name
        field_mask: Series = codebook['field_name'] == field_name
        if sum(form_mask) == 0:
            raise ValueError(f'Form {form_name} not found in codebook')
        if sum(field_mask) == 0:
            raise ValueError(f'Field {field_name} not found in codebook')

        choices = codebook.loc[
            checkbox_mask & form_mask & field_mask,
            'select_choices_or_calculations'].tolist()[0]
        # Subset raw values
        choices = [pair.split(',')[0].strip() for pair in choices.split('|')]

        # Select columns of prefix [field_name]
        cols_to_decode = [field_name + '___' + choice for choice in choices]
        logging.info('Decoding %s', cols_to_decode)

        # Create a copy of the columns to decode
        aux_df = df[cols_to_decode].copy()

        # Find index for the first occurence
        insert_index = df.columns.get_loc(cols_to_decode[0])

        # Remove field_name prefix from column names
        aux_df.columns = aux_df.columns.str.replace(field_name + '___', '')

        # Check if data has different values than boolean ones (0 or 1)
        assert aux_df.isin([0, 1, np.nan]).all().all(), 'Data has values different than 0, 1 or NaN'

        # Replace each value with the respective column name
        decoded_col = aux_df.apply(lambda x: x.index[x == True].values, axis=1)  # Ignore pylint C0121

        # Join values with join_str
        if join_str:
            decoded_col = decoded_col.apply(join_str.join)

        # Set empty list/values to NaN
        decoded_col = decoded_col.apply(lambda x: np.nan if len(x) == 0 else x)

        # Drop original encoded columns
        df = df.drop(columns=cols_to_decode)

        # Add decoded column to dataframe
        df.insert(insert_index, field_name, decoded_col)

        return df

# def _deprecated_create_feature_map(self) -> Series:
#     logging.info('Creating feature map.')
#
#     # Get instruments that doesn't repeat
#     single_instruments = set(self.codebook['form_name']) - set(self.df['redcap_repeat_instrument'].dropna())
#
#     # Get the list of unique features for each form_name
#     feature_map: DataFrame = (
#         self.codebook[['form_name', 'field_name']]
#         .groupby('form_name')
#         .agg(set)
#     )
#
#     # Create a set of all field names from feature_map
#     all_field_names = set.union(*feature_map['field_name'])
#
#     # Find the common columns between df and all_field_names
#     common_columns = self.df.columns.intersection(all_field_names)
#
#     # Update the feature_map DataFrame in one operation
#     feature_map: Series = feature_map['field_name'].apply(lambda x: list(common_columns.intersection(x)))
#
#     # Add identifier fields for each row
#     for form in feature_map.index:
#         if 'record_id' not in feature_map[form]:
#             feature_map[form].insert(0, 'record_id')
#         if 'redcap_data_access_group' not in feature_map[form]:
#             feature_map[form].insert(1, 'redcap_data_access_group')
#     # Add instance_id field for each row
#     for form in feature_map.index:
#         if form not in single_instruments:
#             feature_map[form].insert(2, 'redcap_repeat_instance')
#     # Add checkbox features
#     if self.codebook['field_type'].isin(['checkbox']).any():
#         checkbox_fields: DataFrame = (
#             self.codebook
#             [self.codebook['field_type'] == 'checkbox']
#             [['form_name', 'field_name']]
#         )
#         checkbox_fields: List[tuple] = [row[0:] for row in checkbox_fields.itertuples(index=False)]
#         for form, feature in checkbox_fields:
#             feature_map[form].extend(self.df.filter(regex=f"^{feature}").columns)
#
#         # Remove any duplicated columns inserted via checkbox features
#         def unique_sorted(x: list) -> list:
#             """Apply ```np.unique``` without losing order."""
#             _, indexes = np.unique(x, return_index=True)
#             return [x[index] for index in sorted(indexes)]
#
#         feature_map = feature_map.apply(unique_sorted)

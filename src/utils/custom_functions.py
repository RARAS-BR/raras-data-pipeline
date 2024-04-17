# TODO: Organize functions in the respective classes
import logging
import re
import numpy as np
import pandas as pd


def fix_peso_nascimento(df):
    return (
        df['peso_nascimento']
        .str.replace('.', '', regex=True)  # Remove any decimal point occurrences
        .astype(float, errors='ignore')  # Cast to numeric
        .apply(lambda x: x if x < 5000 else np.nan)  # Remove outliers
        .apply(lambda x: x if x // 10 else x * 1000)  # Multiply by 1000 if it has only 1 digit
        .apply(lambda x: x if x // 100 else x * 100)  # Multiply by 100 if it has only 2 digits
        .astype('Int64')  # Enforce correct dtype
    )


def create_diag_etiologico_df(df: pd.DataFrame) -> pd.DataFrame:

    # Columns mappings
    id_cols = ['record_id', 'redcap_data_access_group', 'redcap_repeat_instance']
    diag_molecular_cols = [
        'tp_diag_etiologico',
        'diag_etio_mol_tipo',
        'outro_tipo_metodo_mol',
        'diag_etio_mol_variante_ident',
        'diag_etio_mol_gene',
        'diag_etio_mol_regiao',
    ]
    diag_citogenico_cols = [
        'tp_diag_etiologico',
        'diag_etio_citogen_tipo',
        'outro_tipo_metodo_cito',
    ]
    diag_bioquimico_cols = [
        'tp_diag_etiologico',
        'diag_etio_erros_inat_met',
        'outro_tipo_metodo_erro_inat',
    ]
    outros_diag_cols = [
        'tp_diag_etiologico',
        'outro_tipo_diagnostico_etio_2',
    ]
    replace_mapping = {
        'redcap_data_access_group': 'id_centro',
        'redcap_repeat_instance': 'instance_id',
        'tp_diag_etiologico': 'tipo_diag_etiologico',
        'diag_etio_mol_tipo': 'metodo',
        'outro_tipo_metodo_mol': 'outro_metodo',
        'diag_etio_erros_inat_met': 'metodo',
        'outro_tipo_metodo_erro_inat': 'outro_metodo',
        'diag_etio_citogen_tipo': 'metodo',
        'outro_tipo_metodo_cito': 'outro_metodo',
        'diag_etio_mol_variante_ident': 'variante',
        'diag_etio_mol_gene': 'gene',
        'diag_etio_mol_regiao': 'regiao',
        'outro_tipo_diagnostico_etio_2': 'outro_tipo_diag_etiologico'
    }

    # Masks
    diag_molecular_mask = df['tp_diag_etiologico'] == 'molecular'
    diag_citogenico_mask = df['tp_diag_etiologico'] == 'citogenetico'
    diag_bioquimico_mask = df['tp_diag_etiologico'] == 'bioquimico'
    outros_diag_mask = ~df['tp_diag_etiologico'].isin(['molecular', 'citogenetico', 'bioquimico'])

    data_frames = []
    for mask, cols in zip([diag_molecular_mask, diag_citogenico_mask, diag_bioquimico_mask, outros_diag_mask],
                          [diag_molecular_cols, diag_citogenico_cols, diag_bioquimico_cols, outros_diag_cols]):
        data = df[mask][id_cols + cols].dropna(how='all')
        data.rename(columns=replace_mapping, inplace=True)
        data_frames.append(data)

    return pd.concat(data_frames).dropna(subset=['tipo_diag_etiologico'])


def extract_ids_and_descriptions(x: str, pattern: str) -> list:
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
        logging.warning(f'No matches found for {x}')
        extracted_id = ''
        extracted_description = ''

    return [extracted_id, extracted_description]


def normalize_orpha_column(df):
    df['doenca_orpha'] = df['doenca_orpha'].dropna().apply(lambda x: ' '.join(x.split()))
    df['doenca_orpha'] = df['doenca_orpha'].str.replace('.0 -', ' -')
    df['doenca_orpha'] = df['doenca_orpha'].dropna().apply(lambda x: x.split('-', 2))

    array_check = df['doenca_orpha'].dropna().apply(len).unique()
    assert len(array_check) == 1 and array_check[0] == 3, 'doenca_orpha column has more/less than 3 elements'

    # Split ORPHA code and description
    df['codigo_orpha'] = df['doenca_orpha'].dropna().apply(lambda x: x[1].strip())
    df['descricao_orpha'] = df['doenca_orpha'].dropna().apply(lambda x: x[2].strip())

    # Insert new columns
    index = df.columns.get_loc('doenca_orpha')
    df.insert(index + 1, 'codigo_orpha', df.pop('codigo_orpha'))
    df.insert(index + 2, 'descricao_orpha', df.pop('descricao_orpha'))

    df.drop('doenca_orpha', axis=1, inplace=True)

    return df

import os
import dotenv
from prefect.blocks.system import Secret, String, JSON
from prefect.filesystems import GitHub

dotenv.load_dotenv()


def main():
    # String variables
    String(value='https://redcap.raras.org.br/api/').save('redcap-api-url', overwrite=True)
    String(value='mongodb://localhost:27017/').save('mongodb-url', overwrite=True)
    String(value='inquerito_retrospectivo_raw').save('retrospectivo-raw-layer', overwrite=True)
    String(value='inquerito_retrospectivo_cleaned').save('retrospectivo-cleaned-layer', overwrite=True)
    String(value='inquerito_prospectivo22_raw').save('prospectivo22-raw-layer', overwrite=True)
    String(value='inquerito_prospectivo22_cleaned').save('prospectivo22-cleaned-layer', overwrite=True)
    String(value='inquerito_prospectivo23_raw').save('prospectivo23-raw-layer', overwrite=True)
    String(value='inquerito_prospectivo23_cleaned').save('prospectivo23-cleaned-layer', overwrite=True)
    String(value='coleta_jav_raw').save('jav-raw-layer', overwrite=True)
    String(value='coleta_jav_cleaned').save('jav-cleaned-layer', overwrite=True)
    String(value='projeto_admin_raw').save('admin-raw-layer', overwrite=True)
    String(value='projeto_admin_cleaned').save('admin-cleaned-layer', overwrite=True)

    # JSON variables
    inquerito_kwargs = {
        "label_columns": [
            "doenca_0k_cid10",
            "doenca_lr_cid10",
            "doenca_sz_cid10",
            "doenca_0k_orpha",
            "doenca_lr_orpha",
            "doenca_sz_orpha",
            "doenca_0k_omim",
            "doenca_lr_omim",
            "doenca_sz_omim"
        ]
    }
    JSON(value=inquerito_kwargs).save('inquerito-kwargs', overwrite=True)

    # Secret variables
    Secret(value=os.getenv('TOKEN_RETROSPECTIVO')).save('token-retrospectivo', overwrite=True)
    Secret(value=os.getenv('TOKEN_PROSPECTIVO_2022')).save('token-prospectivo-2022', overwrite=True)
    Secret(value=os.getenv('TOKEN_PROSPECTIVO_2023')).save('token-prospectivo-2023', overwrite=True)
    Secret(value=os.getenv('TOKEN_JAV')).save('token-jav', overwrite=True)
    Secret(value=os.getenv('TOKEN_ADMIN')).save('token-admin', overwrite=True)
    Secret(value=os.getenv('POSTGRES_USER')).save('postgres-user', overwrite=True)
    Secret(value=os.getenv('POSTGRES_PASSWORD')).save('postgres-password', overwrite=True)

    # Remote repositories
    # TODO: replace with GitHubRepository` block from prefect-github package until September 2024
    GitHub(repository='https://github.com/baiochi/prefect_tutorial.git',
           include_git_objects=True).save('github-repo', overwrite=True)


if __name__ == '__main__':
    main()

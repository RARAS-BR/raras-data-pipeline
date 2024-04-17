from prefect import flow
from prefect.filesystems import GitHub

github_block = GitHub.load("github-repo")

if __name__ == "__main__":
    flow.from_source(
        source=github_block,
        entrypoint="flows/prospectivo22/pipeline_flow.py:run_flow",
    ).deploy(
        name="prospectivo_2022_deploy",
        work_pool_name="default-agent-pool",
    )

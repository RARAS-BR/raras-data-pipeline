from prefect import flow, Flow
from prefect.filesystems import GitHub

github_block = GitHub.load("github-repo")

if __name__ == "__main__":
    flow.from_source(
        source=github_block,
        entrypoint="flows/retrospectivo/pipeline_flow.py:run_flow",
    ).deploy(
        name="retrospectivo_deploy",
        work_pool_name="default-agent-pool",
    )

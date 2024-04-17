from prefect import flow
from prefect.filesystems import GitHub

github_block = GitHub.load("github-repo")

if __name__ == "__main__":
    flow.from_source(
        source=github_block,
        entrypoint="flows/prospectivo23/pipeline_flow.py:run_flow",
    ).deploy(
        name="prospectivo_2023_deploy",
        work_pool_name="default-agent-pool",
        cron="0 23 * * *",  # Run every day at 23:00
    )

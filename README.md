# Prefect Tutorial

```bash
# Install packages
python -m pip install --upgrade pip
pip install -e .
# Start server
prefect server start
# Build from local file
prefect deployment build <FILE_NAME>.py:<FLOW_FUNCTION> -n <DEPLOY_NAME>
# Build from remote repository
prefect deployment build <FILE_NAME>.py:<FLOW_FUNCTION>> -n <DEPLOY_NAME> -sb github/<REPO_NAME>
# Apply deployment
prefect deployment apply <DEPLOY_YAML_FILE>
# Star agent
prefect agent start -q '<AGENT_NAME>'
```

## Example

```bash
prefect server start
prefect deployment build pipeline.py:flow_function -n remote_deploy -sb github/my-remote-repo
prefect deployment apply flow_function-deployment.yaml
prefect agent start -q 'default'
```

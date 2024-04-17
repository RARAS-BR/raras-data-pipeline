
### Workflow example
```bash
# setup Postgres
prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://postgres:yourTopSecretPassword@localhost:5432/prefect"

# Start server  
prefect server start

# Generate blocks
python generate_blocks.py

# Create and start worker  
prefect work-pool create default-agent-pool  
prefect worker start --pool 'default-agent-pool'  

# Deploy flow
python create_deployment.py

# Schedule a imediate run
prefect deployment run 'flow_name/deploy_name'
```

### Inquerito workflow
```bash
python ./src/utils/generate_blocks.py
python ./flows/retrospectivo/create_deployment.py
python ./flows/prospectivo22/create_deployment.py
python ./flows/prospectivo23/create_deployment.py
prefect deployment run 'retrospectivo_flow/retrospectivo_deploy'
prefect deployment run 'prospectivo_2022_flow/prospectivo_2022_deploy'
prefect deployment run 'prospectivo_2023_flow/prospectivo_2023_deploy'
```

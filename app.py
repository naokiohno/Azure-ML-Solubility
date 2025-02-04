
# Import libraries -----------------------------------------------------------------------------------------------------
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Authenticate Azure workspace -----------------------------------------------------------------------------------------
credential = DefaultAzureCredential()

# Specify workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="9afe6e00-d64a-4427-9ff4-4990adeac165",
    resource_group_name="Pay-as-you-go-resource-group",
    workspace_name="Pay-as-you-go-workspace")

# Check successful authentication. If no error, you're good to go.
ws = ml_client.workspaces.get("Pay-as-you-go-workspace")
print(ws.location, ":", ws.resource_group)

# Download data and upload to Blob Storage -----------------------------------------------------------------------------

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

data_path = "data/solubility_full.csv"

solubility_data = Data(
    name="solubility_csv",
    path=data_path,
    type=AssetTypes.URI_FILE,
    description="Dataset for solubility modelling",
    tags={"source_type": "local", "source": "local storage"},
    version="1.1.0",
)

solubility_data = ml_client.data.create_or_update(solubility_data)
print(
    f"Dataset with name {solubility_data.name} was registered to workspace, the dataset version is {solubility_data.version}"
)

# Create job environment -----------------------------------------------------------------------------------------------

dependencies_dir = "./dependencies"

import os
from azure.ai.ml.entities import Environment

custom_env_name = "aml-scikit-learn-solubility"

pipeline_job_env = Environment(
    name=custom_env_name,
    description="Custom environment for Solubility ML pipeline",
    tags={"scikit-learn": "0.24.2"},
    conda_file=os.path.join(dependencies_dir, "conda.yaml"),
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    version="0.1.1",
)
pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

print(
    f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}"
)

# Create training pipeline ---------------------------------------------------------------------------------------------

# Data prep ---------------------------------
data_prep_src_dir = "./components/data_prep"

from azure.ai.ml import command
from azure.ai.ml import Input, Output

data_prep_component = command(
    name="data_prep_solubility",
    display_name="Data preparation for training",
    description="reads a csv input, split the input to train and test, then performs a preprocessing pipeline",
    inputs={
        "data": Input(type="uri_folder"),
        "test_train_ratio": Input(type="number"),
    },
    outputs=dict(
        train_data=Output(type="uri_folder", mode="rw_mount"),
        test_data=Output(type="uri_folder", mode="rw_mount"),
    ),
    # The source folder of the component
    code=data_prep_src_dir,
    command="""python data_prep.py \
            --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} \
            --train_data ${{outputs.train_data}} --test_data ${{outputs.test_data}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

# Now we register the component to the workspace
data_prep_component = ml_client.create_or_update(data_prep_component.component)

# Create (register) the component in your workspace
print(
    f"Component {data_prep_component.name} with Version {data_prep_component.version} is registered"
)

# Model training -----------------------------

train_src_dir = "./components/train"

# importing the Component Package
from azure.ai.ml import load_component

# Loading the component from the yml file
train_component = load_component(source=os.path.join(train_src_dir, "train.yml"))

# Now we register the component to the workspace
train_component = ml_client.create_or_update(train_component)

# Create (register) the component in your workspace
print(
    f"Component {train_component.name} with Version {train_component.version} is registered"
)


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
    version="1.0.0",
)

solubility_data = ml_client.data.create_or_update(solubility_data)
print(
    f"Dataset with name {solubility_data.name} was registered to workspace, the dataset version is {solubility_data.version}"
)


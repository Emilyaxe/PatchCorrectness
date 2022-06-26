import wandb
from wandb.tensorflow import WandbHook
from wandb.apis import InternalApi
wandb.init(name="ocor")
print('Uploading predictions to W&B')
# upload model predictions CSV file to W&B

# we checked that there are three path components above
#entity, project, name = args_wandb_run_id.split('/')

# make sure the file is in our cwd, with the correct name
predictions_base_csv = "model_predictions.csv"
#shutil.copyfile(predictions_csv, predictions_base_csv)

# Using internal wandb API. TODO: Update when available as a public API
internal_api = InternalApi()
internal_api.push([predictions_base_csv], run="v05ida2n", entity="github", project="codesearchnet")

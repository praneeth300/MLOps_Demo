from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="../data/teleco_churn.csv",
    repo_id="praneeth232/test",
    repo_type="dataset",
)

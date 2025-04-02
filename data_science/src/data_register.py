from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="data",
    repo_id="praneeth232/test",
    repo_type="dataset",
)


# from datasets import Dataset
# from huggingface_hub import HfApi
# import os

# # Define variables
# HF_TOKEN = os.getenv("HF_TOKEN")
# DATASET_REPO = "praneeth232/test"

# # Load dataset from CSV
# dataset = Dataset.from_csv("data.csv")

# # Push dataset to Hugging Face Hub
# dataset.push_to_hub(DATASET_REPO, token=HF_TOKEN)

# print(f"Dataset uploaded to: https://huggingface.co/datasets/{DATASET_REPO}")

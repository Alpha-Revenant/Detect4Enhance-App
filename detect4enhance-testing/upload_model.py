from huggingface_hub import HfApi, HfFolder
import os

# Set Hugging Face API Token (can be set in your environment variables or hardcoded)
api_token = "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"  # Replace with your Hugging Face token

# Set the API token for Hugging Face
HfFolder.save_token(api_token)

# Replace with your Hugging Face username
USERNAME = "Phantom-0-0"
REPO_NAME = "detect4enhance-tflite-model"  # Or any name you like

# Create the repo
api = HfApi()
api.create_repo(repo_id=f"{USERNAME}/{REPO_NAME}", exist_ok=True)

# Your model file path
model_file = "engagement_model_89.tflite"

# Upload to Hugging Face
api.upload_file(
    path_or_fileobj=model_file,
    path_in_repo="model.tflite",
    repo_id=f"{USERNAME}/{REPO_NAME}",
    repo_type="model"
)

print("Model uploaded successfully!")

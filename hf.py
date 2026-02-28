"""HuggingFace Hub utilities for dataset and model management."""

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, list_repo_files, snapshot_download


def upload_file(local_path: str, repo_id: str, path_in_repo: str, repo_type: str = "dataset"):
    """Upload a single file to a HuggingFace repo."""
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Uploaded {local_path} -> {repo_id}/{path_in_repo}")


def upload_folder(local_dir: str, repo_id: str, repo_type: str = "dataset"):
    """Upload an entire folder to a HuggingFace repo."""
    api = HfApi()
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Uploaded {local_dir} -> {repo_id}")


def download_model(repo_id: str, local_dir: str = "./models") -> Path:
    """Download a model from HuggingFace Hub."""
    path = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"Downloaded {repo_id} -> {path}")
    return Path(path)


def download_file(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    """Download a single file from a HuggingFace repo."""
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    print(f"Downloaded {repo_id}/{filename} -> {path}")
    return path


def list_files(repo_id: str, repo_type: str = "dataset") -> list[str]:
    """List files in a HuggingFace repo."""
    files = list(list_repo_files(repo_id=repo_id, repo_type=repo_type))
    print(f"Files in {repo_id}: {files}")
    return files


def check_job_status(job_id: str):
    """Check the status of a HuggingFace Job."""
    api = HfApi()
    # HF Jobs status checking - API may vary
    info = api.get_space_runtime(job_id)
    print(f"Job {job_id}: {info}")
    return info


def create_repo(repo_id: str, repo_type: str = "dataset", private: bool = False):
    """Create a new HuggingFace repo."""
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    print(f"Created repo: {repo_id} (type={repo_type})")

from pathlib import Path
from huggingface_hub import hf_hub_download
from .find_directory import source_dir
from .labels_specs import REPO

def fetch(relative_path,local_root, revision):
    local = Path(local_root)/relative_path
    if local.exists():
        return local
    return Path(hf_hub_download(repo_id=REPO,repo_type="dataset",revision=revision,filename=relative_path,local_dir=str(local_root),))


def fetch_ground_truth(out,cases,revision):
    for case in cases:
        fetch(case['label_file'],source_dir(out),revision)
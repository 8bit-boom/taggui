from pathlib import Path

from huggingface_hub import snapshot_download
from PySide6.QtCore import QThread, Signal


class ModelManagerThread(QThread):
    """Downloads a model from Hugging Face Hub in the background."""
    download_finished = Signal(bool, str)

    def __init__(self, parent, model_id: str, local_directory_path: Path):
        super().__init__(parent)
        self.model_id = model_id
        self.local_directory_path = local_directory_path

    def run(self):
        try:
            snapshot_download(repo_id=self.model_id,
                              local_dir=str(self.local_directory_path))
        except Exception as exception:
            self.download_finished.emit(False, str(exception))
            return
        self.download_finished.emit(True, '')

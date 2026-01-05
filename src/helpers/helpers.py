import sys
from pathlib import Path

from PySide6.QtWidgets import QFileDialog


def get_root_dir() -> Path:
    """
    Get the root directory of the __main__ file.

    Returns [str]:
        Path object
    """
    if getattr(sys, 'frozen', False):  # Check if running from the PyInstaller EXE
        return Path(getattr(sys, '_MEIPASS', '.'))
    else:  # Running in a normal Python environment
        return Path(__file__).resolve().parents[1]


def select_folder(default_dir: str | None = None) -> str:
    """
    Open a file dialog to select a folder.

    Returns:
        Path: The path to the selected folder. If the dialog is cancelled,
             an empty string is returned.
    """
    if not default_dir:
        default_dir = ''
    folder_path: str = QFileDialog.getExistingDirectory(
        parent=None,
        caption='Choose Folder',
        dir=default_dir,
        options=QFileDialog.Option.ShowDirsOnly,
    )
    return folder_path


def select_file(default_dir: str | None = None) -> str:
    file_path: str
    if not default_dir:
        default_dir = ''
    file_path, _ = QFileDialog.getOpenFileName(
        parent=None, caption='Choose File', dir=default_dir
    )
    return file_path


if __name__ == '__main__':
    root_dir = get_root_dir()
    print(root_dir)

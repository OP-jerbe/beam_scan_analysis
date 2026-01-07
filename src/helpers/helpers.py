import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from plotly.graph_objects import Figure
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
        str: The path to the selected folder. If the dialog is cancelled,
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


def get_save_filename(default_filename: str | None = None) -> str:
    filename: str
    if not default_filename:
        default_filename = ''
    filename, _ = QFileDialog.getSaveFileName(
        parent=None,
        dir=default_filename,
        caption='Save CSV File',
        filter='CSV Files (*.csv);;All Files (*)',
    )
    return filename


def save_all_as_html(
    folder_path: str,
    titles: list[str],
    figs: list[Figure | None],
) -> None:
    folder = Path(folder_path)

    for title, fig in zip(titles, figs):
        file_name = f'{title}'
        full_path = folder / file_name
        if fig:
            fig.write_html(str(full_path))


def get_app_version() -> str:
    try:
        return version('beam_scan_analysis')
    except PackageNotFoundError:
        return 'development-build'


if __name__ == '__main__':
    root_dir = get_root_dir()
    print(root_dir)

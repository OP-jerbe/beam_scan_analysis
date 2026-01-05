from PySide6.QtCore import QObject, Slot

from ..model.model import Model
from ..view.main_window import MainWindow


class Controller(QObject):
    def __init__(self, model: Model, view: MainWindow) -> None:
        super().__init__()
        self.model = model
        self.view = view

        self.view.load_scan_data_sig.connect(self.receive_select_csv_file_sig)
        self.view.plot_beam_scan_sig.connect(self.receive_plot_beam_scan_sig)

    @Slot()
    def receive_select_csv_file_sig(self, filepath: str) -> None:
        self.model.load_scan_data(filepath)

    @Slot()
    def receive_create_grid_sig(self) -> None:
        self.model.create_grid()

    @Slot()
    def receive_plot_beam_scan_sig(self) -> None:
        print('Received plot_beam_scan_sig')

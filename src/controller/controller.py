from PySide6.QtCore import QObject, Slot

from ..model.model import Model
from ..view.beam_scan_plotting import Heatmap, IPrime, Surface, XYCrossSections
from ..view.main_window import MainWindow


class Controller(QObject):
    def __init__(self, model: Model, view: MainWindow) -> None:
        super().__init__()
        self.model = model
        self.view = view

        self.view.load_scan_data_sig.connect(self.receive_select_csv_file_sig)
        self.view.plot_3d_surface_sig.connect(self.receive_plot_3d_surface_sig)
        self.view.plot_heatmap_sig.connect(self.receive_plot_heatmap_sig)
        self.view.plot_xy_cross_sections_sig.connect(
            self.receive_plot_xy_cross_sections_sig
        )
        self.view.plot_i_prime_sig.connect(self.receive_plot_i_prime_sig)

    @Slot()
    def receive_select_csv_file_sig(self, filepath: str) -> None:
        self.model.load_scan_data(filepath)

    @Slot()
    def receive_create_grid_sig(self) -> None:
        self.model.create_grid()

    @Slot()
    def receive_plot_3d_surface_sig(
        self, diam: float, dist: float, z_scale: list
    ) -> None:
        surface = Surface(self.model.bs, diam, dist, z_scale)
        surface.plot_surface()

    @Slot()
    def receive_plot_heatmap_sig(self, diam: float, dist: float, z_scale: list) -> None:
        heatmap = Heatmap(self.model.bs, diam, dist, z_scale)
        heatmap.plot_heatmap()

    @Slot()
    def receive_plot_xy_cross_sections_sig(
        self, diam: float, dist: float, z_scale: list
    ) -> None:
        xy_cross_sections = XYCrossSections(self.model.bs, diam, dist, z_scale)
        xy_cross_sections.plot_cross_sections()

    @Slot()
    def receive_plot_i_prime_sig(self, diam: float, dist: float) -> None:
        i_prime = IPrime(self.model.bs, diam, dist)
        i_prime.plot_i_prime()

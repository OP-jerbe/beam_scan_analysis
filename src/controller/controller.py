from PySide6.QtCore import QObject, QThreadPool, Slot

from ..model.model import Model
from ..model.worker import Worker
from ..view.beam_scan_plotting import Heatmap, IPrime, Surface, XYCrossSections
from ..view.main_window import MainWindow


class Controller(QObject):
    def __init__(self, model: Model, view: MainWindow) -> None:
        super().__init__()
        self.model = model
        self.view = view
        self.thread_pool = QThreadPool()

        self.view.load_scan_data_sig.connect(self.receive_select_csv_file_sig)
        self.view.plot_3d_surface_sig.connect(self.receive_plot_3d_surface_sig)
        self.view.plot_heatmap_sig.connect(self.receive_plot_heatmap_sig)
        self.view.plot_xy_cross_sections_sig.connect(
            self.receive_plot_xy_cross_sections_sig
        )
        self.view.plot_i_prime_sig.connect(self.receive_plot_i_prime_sig)

    def _run_plot_worker(self, func, error_handler) -> None:
        worker = Worker(func)
        worker.signals.error.connect(error_handler)
        self.thread_pool.start(worker)

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
        self._run_plot_worker(surface.plot, self.view.surface_error_message)

    @Slot()
    def receive_plot_heatmap_sig(self, diam: float, dist: float, z_scale: list) -> None:
        heatmap = Heatmap(self.model.bs, diam, dist, z_scale)
        self._run_plot_worker(heatmap.plot, self.view.heatmap_error_message)

    @Slot()
    def receive_plot_xy_cross_sections_sig(
        self, diam: float, dist: float, z_scale: list
    ) -> None:
        xy_cross_sections = XYCrossSections(self.model.bs, diam, dist, z_scale)
        self._run_plot_worker(
            xy_cross_sections.plot,
            self.view.cross_sections_error_message,
        )

    @Slot()
    def receive_plot_i_prime_sig(self, diam: float, dist: float) -> None:
        i_prime = IPrime(self.model.bs, diam, dist)
        self._run_plot_worker(i_prime.plot, self.view.i_prime_error_message)

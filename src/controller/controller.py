from plotly.graph_objects import Figure
from PySide6.QtCore import QObject, QThreadPool, Slot

import src.helpers.helpers as h

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
        self.graphs: list[Heatmap | IPrime | Surface | XYCrossSections] = []

        self.view.load_scan_data_sig.connect(self.receive_select_csv_file_sig)
        self.view.plot_3d_surface_sig.connect(self.receive_plot_3d_surface_sig)
        self.view.plot_heatmap_sig.connect(self.receive_plot_heatmap_sig)
        self.view.plot_xy_cross_sections_sig.connect(
            self.receive_plot_xy_cross_sections_sig
        )
        self.view.plot_i_prime_sig.connect(self.receive_plot_i_prime_sig)
        self.view.export_to_csv_sig.connect(self.receive_export_to_csv_sig)
        self.view.disable_interp_sig.connect(self.receive_disable_interp_sig)
        self.view.save_html_figure_sig.connect(self.receive_save_html_figure_sig)

    def _run_plot_worker(self, func, error_handler) -> None:
        worker = Worker(func)
        worker.signals.error.connect(error_handler)
        self.thread_pool.start(worker)

    def _run_save_html_plot_worker(self, func, error_handler, *args, **kwargs) -> None:
        worker = Worker(func, rtn=True, *args, **kwargs)
        worker.signals.error.connect(error_handler)
        worker.signals.rtn.connect(self.receive_worker_rtn_sig)
        self.thread_pool.start(worker)

    @Slot()
    def receive_select_csv_file_sig(self, filepath: str) -> None:
        self.model.load_scan_data(filepath)

    @Slot()
    def receive_create_grid_sig(self) -> None:
        self.model.create_grid()

    @Slot()
    def receive_plot_3d_surface_sig(self, inputs: dict, z_scale: list) -> None:
        surface = Surface(self.model.bs, inputs, z_scale)
        self._run_plot_worker(surface.plot, self.view.surface_error_message)

    @Slot()
    def receive_plot_heatmap_sig(self, inputs: dict, z_scale: list) -> None:
        heatmap = Heatmap(self.model.bs, inputs, z_scale)
        self._run_plot_worker(heatmap.plot, self.view.heatmap_error_message)

    @Slot()
    def receive_plot_xy_cross_sections_sig(self, inputs: dict, z_scale: list) -> None:
        xy_cross_sections = XYCrossSections(self.model.bs, inputs, z_scale)
        self._run_plot_worker(
            xy_cross_sections.plot,
            self.view.cross_sections_error_message,
        )

    @Slot()
    def receive_plot_i_prime_sig(self, inputs: dict) -> None:
        i_prime = IPrime(self.model.bs, inputs)
        self._run_plot_worker(i_prime.plot, self.view.i_prime_error_message)

    @Slot()
    def receive_export_to_csv_sig(self, filename: str, inputs: dict) -> None:
        worker = Worker(self.model.export_to_csv, filename=filename, inputs=inputs)
        worker.signals.error.connect(self.view.csv_export_error_message)
        self.thread_pool.start(worker)

    @Slot()
    def receive_disable_interp_sig(self, checked: bool) -> None:
        if checked:
            self.model.create_grid(interp_num=None)
        else:
            self.model.create_grid(interp_num=self.model.bs.interp_num)

    @Slot()
    def receive_save_html_figure_sig(
        self, which: str, inputs: dict, z_scale: list, folder_path: str
    ) -> None:
        match which:
            case 'surface':
                self.graphs = [Surface(self.model.bs, inputs, z_scale)]
                error_handler = self.view.surface_error_message
            case 'heatmap':
                self.graphs = [Heatmap(self.model.bs, inputs, z_scale)]
                error_handler = self.view.heatmap_error_message
            case 'xy_cross_section':
                self.graphs = [XYCrossSections(self.model.bs, inputs, z_scale)]
                error_handler = self.view.cross_sections_error_message
            case 'i_prime':
                self.graphs = [IPrime(self.model.bs, inputs)]
                error_handler = self.view.i_prime_error_message
            case 'all':
                self.graphs = [
                    Heatmap(self.model.bs, inputs, z_scale),
                    IPrime(self.model.bs, inputs),
                    Surface(self.model.bs, inputs, z_scale),
                    XYCrossSections(self.model.bs, inputs, z_scale),
                ]
                error_handler = self.view.save_html_error_message
            case _:
                error_handler = self.view.save_html_error_message
                return

        if which != 'all':
            self._run_save_html_plot_worker(
                self.graphs[0].plot,
                show=False,
                error_handler=error_handler,
            )
        else:
            titles: list[str] = [
                'heatmap.html',
                'ang_int_vs_div.html',
                'surface.html',
                'xy_cross_section.html',
            ]
            figs = self._get_all_figures(self.graphs)
            h.save_all_as_html(folder_path, titles, figs)

    @staticmethod
    def _get_all_figures(
        graphs: list[Heatmap | IPrime | Surface | XYCrossSections],
    ) -> list[Figure | None]:
        figs: list[Figure | None] = []
        for graph in graphs:
            fig = graph.plot(show=False)
            if fig:
                figs.append(fig)
        return figs

    @Slot()
    def receive_worker_rtn_sig(self, obj) -> None:
        """Runs when the `_run_save_plot_worker` method finishes successfully."""
        self.graphs[0].save_as_html(obj)

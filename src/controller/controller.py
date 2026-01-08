from plotly.graph_objects import Figure
from PySide6.QtCore import QObject, QThreadPool, Signal, Slot

import src.helpers.helpers as h

from ..model.model import Model
from ..model.worker import Worker
from ..view.beam_scan_plotting import Heatmap, IPrime, Surface, XYCrossSections
from ..view.main_window import MainWindow


class Controller(QObject):
    save_single_html_file_sig = Signal(object)
    save_all_html_files_sig = Signal(object)
    save_single_png_file_sig = Signal(object)
    save_all_png_files_sig = Signal(object)

    def __init__(self, model: Model, view: MainWindow) -> None:
        super().__init__()
        self.model = model
        self.view = view
        self.thread_pool = QThreadPool()
        self.graphs: list[Heatmap | IPrime | Surface | XYCrossSections] = []
        self.folder_path: str = ''
        self.filepath: str = ''
        self.save_file_type: str = ''
        self.titles: list[str] = []

        # --- Internal Controller Signal Connections ---

        self.save_single_html_file_sig.connect(self.receive_save_single_html_file_sig)
        self.save_all_html_files_sig.connect(self.receive_save_all_html_files_sig)
        self.save_single_png_file_sig.connect(self.receive_save_single_png_file_sig)
        self.save_all_png_files_sig.connect(self.receive_save_all_png_files_sig)

        # --- View Signal Connections ---

        self.view.load_scan_data_sig.connect(self.receive_load_scan_data_sig)
        self.view.plot_3d_surface_sig.connect(self.receive_plot_3d_surface_sig)
        self.view.plot_heatmap_sig.connect(self.receive_plot_heatmap_sig)
        self.view.plot_xy_cross_sections_sig.connect(
            self.receive_plot_xy_cross_sections_sig
        )
        self.view.plot_i_prime_sig.connect(self.receive_plot_i_prime_sig)
        self.view.export_to_csv_sig.connect(self.receive_export_to_csv_sig)
        self.view.disable_interp_sig.connect(self.receive_disable_interp_sig)
        self.view.save_html_figure_sig.connect(self.receive_save_html_figure_sig)
        self.view.save_png_figure_sig.connect(self.receive_save_png_figure_sig)
        self.view.folder_path_sig.connect(self.receive_folder_path_sig)
        self.view.filename_sig.connect(self.receive_filepath_sig)
        self.view.file_type_sig.connect(self.receive_file_type_sig)
        self.view.titles_sig.connect(self.receive_titles_sig)

    # --- Load Scan Data ---

    @Slot()
    def receive_load_scan_data_sig(self, filepath: str) -> None:
        self.model.load_scan_data(filepath)

    # --- Disable Interpolation ---

    @Slot()
    def receive_disable_interp_sig(self, checked: bool) -> None:
        if checked:
            self.model.create_grid(interp_num=None)
        else:
            self.model.create_grid(interp_num=self.model.bs.interp_num)

    # --- Export to data to csv ---

    @Slot()
    def receive_export_to_csv_sig(self, filename: str, inputs: dict) -> None:
        worker = Worker(self.model.export_to_csv, filename=filename, inputs=inputs)
        worker.signals.error.connect(self.view.csv_export_error_message)
        self.thread_pool.start(worker)

    # --- Plotting ---

    def _run_plotting_worker(self, func, error_handler) -> None:
        worker = Worker(func)
        worker.signals.error.connect(error_handler)
        self.thread_pool.start(worker)

    @Slot()
    def receive_plot_3d_surface_sig(self, inputs: dict) -> None:
        surface = Surface(self.model.bs, inputs)
        self._run_plotting_worker(surface.plot, self.view.surface_error_message)

    @Slot()
    def receive_plot_heatmap_sig(self, inputs: dict) -> None:
        heatmap = Heatmap(self.model.bs, inputs)
        self._run_plotting_worker(heatmap.plot, self.view.heatmap_error_message)

    @Slot()
    def receive_plot_xy_cross_sections_sig(self, inputs: dict) -> None:
        xy_cross_sections = XYCrossSections(self.model.bs, inputs)
        self._run_plotting_worker(
            xy_cross_sections.plot,
            self.view.cross_sections_error_message,
        )

    @Slot()
    def receive_plot_i_prime_sig(self, inputs: dict) -> None:
        i_prime = IPrime(self.model.bs, inputs)
        self._run_plotting_worker(i_prime.plot, self.view.i_prime_error_message)

    # --- Save helpers ---

    def _run_figure_rtn_worker(self, func, error_handler, *args, **kwargs) -> None:
        worker = Worker(func, rtn=True, *args, **kwargs)
        worker.signals.error.connect(error_handler)
        worker.signals.rtn.connect(self.receive_worker_rtn_sig)
        self.thread_pool.start(worker)

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
    def receive_folder_path_sig(self, folder_path: str) -> None:
        self.folder_path = folder_path

    @Slot()
    def receive_filepath_sig(self, filepath: str) -> None:
        self.filepath = filepath

    @Slot()
    def receive_file_type_sig(self, file_type: str) -> None:
        self.save_file_type = file_type

    @Slot()
    def receive_titles_sig(self, titles: list[str]) -> None:
        self.titles = titles

    @Slot()
    def receive_worker_rtn_sig(self, obj) -> None:
        """Runs when the `_run_figure_rtn_worker` method finishes successfully."""
        match self.save_file_type:
            case 'html':
                if len(self.graphs) == 1:
                    self.save_single_html_file_sig.emit(obj)  # list of one figure
                else:
                    self.save_all_html_files_sig.emit(obj)  # list of all figures
            case 'png':
                if len(self.graphs) == 1:
                    self.save_single_png_file_sig.emit(obj)  # list of one figure
                else:
                    self.save_all_png_files_sig.emit(obj)  # list of all figures
            case _:
                raise ValueError('Must give filetype: "html" or "png"')

    # --- Save as HTML ---

    @Slot()
    def receive_save_single_html_file_sig(self, fig) -> None:
        worker = Worker(h.save_as_html, filepath=self.filepath, fig=fig)
        worker.signals.error.connect(self.view.save_html_error_message)
        self.thread_pool.start(worker)

    @Slot()
    def receive_save_all_html_files_sig(self, figs) -> None:
        worker = Worker(
            h.save_all_as_html,
            folder_path=self.folder_path,
            titles=self.titles,
            figs=figs,
        )
        worker.signals.error.connect(self.view.save_html_error_message)
        self.thread_pool.start(worker)

    @Slot()
    def receive_save_html_figure_sig(self, which: str, inputs: dict) -> None:
        match which:
            case 'surface':
                self.graphs = [Surface(self.model.bs, inputs)]
                error_handler = self.view.surface_error_message
            case 'heatmap':
                self.graphs = [Heatmap(self.model.bs, inputs)]
                error_handler = self.view.heatmap_error_message
            case 'xy_cross_section':
                self.graphs = [XYCrossSections(self.model.bs, inputs)]
                error_handler = self.view.cross_sections_error_message
            case 'i_prime':
                self.graphs = [IPrime(self.model.bs, inputs)]
                error_handler = self.view.i_prime_error_message
            case 'all':
                self.graphs = [
                    Heatmap(self.model.bs, inputs),
                    IPrime(self.model.bs, inputs),
                    Surface(self.model.bs, inputs),
                    XYCrossSections(self.model.bs, inputs),
                ]
                error_handler = self.view.save_html_error_message
            case _:
                error_handler = self.view.save_html_error_message

        if which != 'all':
            graph = self.graphs[0]
            self._run_figure_rtn_worker(
                graph.plot,
                show=False,
                error_handler=error_handler,
            )
        else:
            self._run_figure_rtn_worker(
                self._get_all_figures, graphs=self.graphs, error_handler=error_handler
            )

    # --- Save as PNG ---

    @Slot()
    def receive_save_png_figure_sig(self, which: str, inputs: dict) -> None:
        match which:
            case 'surface':
                ...
            case 'heatmap':
                ...
            case 'xy_cross_section':
                ...
            case 'i_prime':
                ...
            case 'all':
                self.graphs = [
                    Heatmap(self.model.bs, inputs),
                    IPrime(self.model.bs, inputs),
                    Surface(self.model.bs, inputs),
                    XYCrossSections(self.model.bs, inputs),
                ]
            case _:
                ...
        error_handler = self.view.save_png_error_message
        if which != 'all':
            ...
        else:
            self._run_figure_rtn_worker(
                self._get_all_figures, graphs=self.graphs, error_handler=error_handler
            )

    @Slot()
    def receive_save_single_png_file_sig(self, fig) -> None:
        worker = Worker(h.save_as_png, filepath=self.filepath, fig=fig)
        worker.signals.error.connect(self.view.save_html_error_message)
        self.thread_pool.start(worker)

    @Slot()
    def receive_save_all_png_files_sig(self, figs) -> None:
        worker = Worker(
            h.save_all_as_png,
            folder_path=self.folder_path,
            titles=self.titles,
            figs=figs,
        )
        worker.signals.error.connect(self.view.save_html_error_message)
        self.thread_pool.start(worker)

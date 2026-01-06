from PySide6.QtCore import QRegularExpression, Signal
from PySide6.QtGui import QRegularExpressionValidator, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class OverrideCentroidWindow(QMainWindow):
    centroid_coords_sig = Signal(list)
    window_closed_without_input_sig = Signal()

    def __init__(self, parent, centroid_coords: list[float]) -> None:
        super().__init__(parent)
        self.submitted = False
        self.setWindowTitle('Override Centroid')
        self.setFixedSize(300, 100)

        # Create the validator for numerical inputs
        number_regex = QRegularExpression(r'^-?\d*\.?\d*$')
        validator = QRegularExpressionValidator(number_regex)

        # Create the input field and label
        override_Xc_label = QLabel('X Coordinate')
        override_Xc_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.override_Xc_input = QLineEdit(f'{centroid_coords[0]}')
        self.override_Xc_input.setFixedHeight(28)
        self.override_Xc_input.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.override_Xc_input.setValidator(validator)

        override_Yc_label = QLabel('Y Coordinate')
        override_Yc_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.override_Yc_input = QLineEdit(f'{centroid_coords[1]}')
        self.override_Yc_input.setFixedHeight(28)
        self.override_Yc_input.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.override_Yc_input.setValidator(validator)

        # Create the button to apply the override
        self.override_button = QPushButton('Set Centroid Coordinates')
        self.override_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.override_button.setFixedHeight(28)
        self.override_button.clicked.connect(self.handle_submit)
        self.override_button.setAutoDefault(True)

        # Arrange widgets in window
        main_layout = QVBoxLayout()
        label_layout = QHBoxLayout()
        input_layout = QHBoxLayout()
        label_layout.addWidget(override_Xc_label)
        label_layout.addWidget(override_Yc_label)
        input_layout.addWidget(self.override_Xc_input)
        input_layout.addWidget(self.override_Yc_input)
        main_layout.addLayout(label_layout)
        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.override_button)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def get_override_values(self) -> list[float] | None:
        x_text = self.override_Xc_input.text()
        y_text = self.override_Yc_input.text()

        if not x_text or not y_text:
            return None

        x_val = float(x_text)
        y_val = float(y_text)
        return [x_val, y_val]

    def handle_submit(self) -> None:
        values = self.get_override_values()
        if values:
            self.centroid_coords_sig.emit(values)
            self.submitted = True
            self.close()
        else:
            QMessageBox.warning(
                self,
                'Warning',
                'Please enter valid coordinates for the centroid.',
            )

    def closeEvent(self, event) -> None:
        if not self.submitted:
            self.window_closed_without_input_sig.emit()
        super().closeEvent(event)

import sys

from PySide6.QtCore import QRegularExpression, Signal
from PySide6.QtGui import QIcon, QRegularExpressionValidator, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from qt_material import apply_stylesheet


class OverrideCentroidWindow(QWidget):
    centroid_set = Signal(float, float)
    window_closed_without_input = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.submitted = False
        self.setWindowTitle('Override Centroid')
        self.setFixedSize(300, 100)
        if hasattr(sys, 'frozen'):  # Check if running from the bundled app
            icon_path = sys._MEIPASS + '/scan.ico'  # type: ignore
        else:
            icon_path = 'scan.ico'  # Use the local icon file in dev mode
        self.setWindowIcon(QIcon(icon_path))
        apply_stylesheet(self, theme='dark_lightgreen.xml', invert_secondary=True)
        self.setStyleSheet(
            self.styleSheet() + """QLineEdit, QTextEdit {color: lightgreen;}"""
        )

        # Create the validator for numerical inputs
        number_regex = QRegularExpression(r'^-?\d*\.?\d*$')
        validator = QRegularExpressionValidator(number_regex)

        # Create the input field and label
        override_Xc_label = QLabel('X Coordinate')
        override_Xc_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.override_Xc_input = QLineEdit()
        self.override_Xc_input.setFixedHeight(28)
        self.override_Xc_input.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.override_Xc_input.setValidator(validator)

        override_Yc_label = QLabel('Y Coordinate')
        override_Yc_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.override_Yc_input = QLineEdit()
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

        self.setLayout(main_layout)

    def get_override_values(self) -> tuple[float, float] | None:
        x_text = self.override_Xc_input.text()
        y_text = self.override_Yc_input.text()

        if not x_text or not y_text:
            return None

        x_val = float(x_text)
        y_val = float(y_text)
        return x_val, y_val

    def handle_submit(self) -> None:
        values = self.get_override_values()
        if values:
            x, y = values
            self.centroid_set.emit(x, y)
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
            self.window_closed_without_input.emit()
        super().closeEvent(event)

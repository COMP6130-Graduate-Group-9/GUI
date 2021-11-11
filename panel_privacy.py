from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QGroupBox, QLabel, QVBoxLayout)

class PanelPrivacy(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, parent=parent)

        self.p = None  # Default empty value.

        titleLabel = QLabel("Privacy")
        titleFont = QFont()
        titleFont.setPointSize(24)
        titleLabel.setFont(titleFont)

        layout = QVBoxLayout(self)
        layout.addWidget(titleLabel)
        self.setLayout(layout)
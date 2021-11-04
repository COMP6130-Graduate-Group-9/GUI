from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QGroupBox, QLabel, QVBoxLayout)

class PanelEffectiveness(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, parent=parent)

        titleLabel = QLabel("Title")
        titleFont = QFont()
        titleFont.setPointSize(24);
        titleLabel.setFont(titleFont)

        layout = QVBoxLayout(self)
        layout.addWidget(titleLabel)
        self.setLayout(layout)
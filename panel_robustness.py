from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QGroupBox, QLabel, QStackedLayout, QVBoxLayout)

from robustness import backdoor, data_poisoning

class PanelRobustness(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, parent=parent)

        self.type_index = {
            "Backdoor Attack": 0,
            "Data Poisoning Attacks": 1
        }

        titleLabel = QLabel("Robustness")
        titleFont = QFont()
        titleFont.setPointSize(20)
        titleLabel.setFont(titleFont)

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.type_switcher = QStackedLayout()
        backdoor_container = backdoor.Container()
        data_poisoning_container = data_poisoning.Container()
        self.type_switcher.addWidget(backdoor_container)
        self.type_switcher.addWidget(data_poisoning_container)
        self.type_switcher.setCurrentIndex(0)
        
        layout.addWidget(titleLabel)
        layout.addLayout(self.type_switcher)
    
    def switch_type(self, name):
        self.type_switcher.setCurrentIndex(self.type_index[name])
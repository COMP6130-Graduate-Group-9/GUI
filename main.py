from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QDialog, QGroupBox, QRadioButton, QComboBox,
    QLabel, QGridLayout, QHBoxLayout)

from panel_effectiveness import PanelEffectiveness

import sys

class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("COMP6130 Graduate Group 9 Project")

        titleLabel = QLabel("Trustworthy Federated Learning")
        titleFont = QFont()
        titleFont.setPointSize(24)
        titleLabel.setFont(titleFont)

        self.createPanelTabs()
        topLayout = QHBoxLayout()
        topLayout.addWidget(titleLabel)
        topLayout.addWidget(self.panelTabs)

        panel = PanelEffectiveness()

        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0)
        mainLayout.addWidget(panel, 1, 0)
        self.setLayout(mainLayout)
    
    def createPanelTabs(self):
        self.panelTabs = QGroupBox("")

        radioButtonEffectiveness = QRadioButton("Effectiveness")
        radioButtonPrivacy = QRadioButton("Privacy")
        radioButtonRobustness = QRadioButton("Robustness")
        radioButtonFairness = QRadioButton("Fairness")
        radioButtonEffectiveness.setChecked(True)

        robustnessComboBox = QComboBox()
        robustnessComboBox.addItems(["Backdoor Attack"])
        robustnessLabel = QLabel("&Type:")
        robustnessLabel.setBuddy(robustnessComboBox)
        robustnessWrapper = QHBoxLayout()
        robustnessWrapper.addWidget(radioButtonRobustness)
        robustnessWrapper.addWidget(robustnessLabel)
        robustnessWrapper.addWidget(robustnessComboBox)

        layout = QGridLayout()
        layout.addWidget(radioButtonEffectiveness, 0, 0)
        layout.addWidget(radioButtonPrivacy, 1, 0)
        layout.addLayout(robustnessWrapper, 0, 1)
        layout.addWidget(radioButtonFairness, 1, 1)
        self.panelTabs.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myapp = MainWindow()
    myapp.show()
    sys.exit(app.exec_())
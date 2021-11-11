from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QDialog, QGroupBox, QRadioButton, QComboBox,
    QLabel, QGridLayout, QHBoxLayout, QStackedLayout)

from panel_effectiveness import PanelEffectiveness
from panel_privacy import PanelPrivacy
from panel_robustness import PanelRobustness
from panel_fairness import PanelFairness

import sys

class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("COMP6130 Graduate Group 9 Project")

        titleLabel = QLabel("Trustworthy Federated Learning")
        titleFont = QFont()
        titleFont.setPointSize(24)
        titleLabel.setFont(titleFont)

        self.createPanelDisplay()
        self.createPanelTabs()
        topLayout = QHBoxLayout()
        topLayout.addWidget(titleLabel)
        topLayout.addWidget(self.panelTabs)

        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0)
        mainLayout.addLayout(self.panelDisplay, 1, 0)
        self.setLayout(mainLayout)
    
    def createPanelTabs(self):
        self.panelTabs = QGroupBox("")

        radioButtonEffectiveness = QRadioButton("Effectiveness")
        radioButtonEffectiveness.toggled.connect(lambda: self.switch_panel(0))
        radioButtonPrivacy = QRadioButton("Privacy")
        radioButtonPrivacy.toggled.connect(lambda: self.switch_panel(1))
        radioButtonRobustness = QRadioButton("Robustness")
        radioButtonRobustness.toggled.connect(lambda: self.switch_panel(2))
        radioButtonFairness = QRadioButton("Fairness")
        radioButtonFairness.toggled.connect(lambda: self.switch_panel(3))
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

    def createPanelDisplay(self):
        self.panelDisplay = QStackedLayout()
        panelEffectiveness = PanelEffectiveness()
        panelPrivacy = PanelPrivacy()
        panelRobustness = PanelRobustness()
        panelFairness = PanelFairness()
        self.panelDisplay.addWidget(panelEffectiveness)
        self.panelDisplay.addWidget(panelPrivacy)
        self.panelDisplay.addWidget(panelRobustness)
        self.panelDisplay.addWidget(panelFairness)
        self.panelDisplay.setCurrentIndex(0)

    def switch_panel(self, index):
        self.panelDisplay.setCurrentIndex(index)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myapp = MainWindow()
    myapp.show()
    sys.exit(app.exec_())
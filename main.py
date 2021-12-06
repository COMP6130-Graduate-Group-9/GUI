from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QAction, QApplication, QDialog, QGroupBox, QRadioButton, QComboBox,
    QLabel, QGridLayout, QHBoxLayout, QStackedLayout, QWidget)

from PyQt5.QtCore import (QFile)
from PyQt5 import QtWidgets

from panel_effectiveness import PanelEffectiveness
from panel_privacy import PanelPrivacy
from panel_robustness import PanelRobustness
from panel_fairness import PanelFairness

import sys
import os

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

        topLayoutWrapper = QWidget()
        topLayoutWrapper.setObjectName("topLayoutWrapper")
        topLayoutWrapper.setLayout(topLayout)

        mainLayout = QGridLayout()
        mainLayout.addWidget(topLayoutWrapper, 0, 0)
        mainLayout.addLayout(self.panelDisplay, 1, 0)
        self.setLayout(mainLayout)
    
    def createPanelTabs(self):
        self.panelTabs = QGroupBox("")
        self.panelTabs.setObjectName("panelTabs")
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
        robustnessComboBox.addItems(
            [
                "Backdoor Attack",
                "Data Poisoning Attacks",
                "Model Poisoning Attacks",
                "Free-rider Attacks",
                "Inference Attacks"
            ]
        )
        robustnessComboBox.currentTextChanged.connect(self.panelRobustness.switch_type)
        # round border for dropdown
        robustnessComboBox.setStyleSheet(
            "border-top-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 #c1c9cf, stop:1 #d2d8dd);"
            "border-right-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #c1c9cf, stop:1 #d2d8dd);"
            "border-bottom-color: qlineargradient(spread:pad, x1:0.5, y1:0, x2:0.5, y2:1, stop:0 #c1c9cf, stop:1 #d2d8dd);"
            "border-left-color: qlineargradient(spread:pad, x1:1, y1:0, x2:0, y2:0, stop:0 #c1c9cf, stop:1 #d2d8dd);"
        )
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
        panelEffectiveness.setObjectName("panelEffectiveness")
        panelPrivacy = PanelPrivacy()
        panelPrivacy.setObjectName("panelPrivacy")
        self.panelRobustness = PanelRobustness()
        self.panelRobustness.setObjectName("panelRobustness")
        panelFairness = PanelFairness()
        panelFairness.setObjectName("panelFairness")
        self.panelDisplay.addWidget(panelEffectiveness)
        self.panelDisplay.addWidget(panelPrivacy)
        self.panelDisplay.addWidget(self.panelRobustness)
        self.panelDisplay.addWidget(panelFairness)
        self.panelDisplay.setCurrentIndex(0)

    def switch_panel(self, index):
        self.panelDisplay.setCurrentIndex(index)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    #app.setStyleSheet('QMainWindow{background-color: darkgray;border: 1px solid black;}')
    # qss
    File = open("qss/MacOS.qss",'r')
    with File:
        qss = File.read()
        app.setStyleSheet(qss)
    myapp = MainWindow()
    myapp.show()
    sys.exit(app.exec_())
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
        self.setStyleSheet("background-color: #151a1e")
        # dark 525252 / mac,aqua ececec / manjaro 151a1e
        titleLabel = QLabel("Trustworthy Federated Learning")
        # titleLabel.setStyleSheet("color: white;"
        #                     "background-color: #496e9c;"
        #                     "border-style: solid;"
        #                     "border-width: 1px;"
        #                     "border-color: white;"
        #                     "border-radius: 1px")

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
        # self.panelTabs.setStyleSheet("color: white;"
        #                     "background-color: #496e9c;"
        #                     "border-style: solid;"
        #                     "border-width: 1px;"
        #                     "border-color: white;"
        #                     "border-radius: 1px")

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
    # qss
    File = open("manjaro.qss",'r')
    with File:
        qss = File.read()
        app.setStyleSheet(qss)
    myapp = MainWindow()
    myapp.show()
    sys.exit(app.exec_())
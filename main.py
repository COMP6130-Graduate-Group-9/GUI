from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QDialog, QGroupBox, QRadioButton, QComboBox,
    QLabel, QVBoxLayout, QGridLayout)

import sys

class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("COMP6130 Graduate Group 9 Project")

        self.createSidebar()
        self.createPanel()

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.sidebar, 0, 0)
        mainLayout.addWidget(self.panel, 0, 1)
        self.setLayout(mainLayout)
    
    def createSidebar(self):
        self.sidebar = QGroupBox("")

        radioButton1 = QRadioButton("Effectiveness")
        radioButton2 = QRadioButton("Privacy")
        radioButton3 = QRadioButton("Robustness")
        radioButton4 = QRadioButton("Fairness")
        radioButton1.setChecked(True)

        robustnessComboBox = QComboBox()
        robustnessComboBox.addItems(["Backdoor Attack"])

        robustnessLabel = QLabel("&Type:")
        robustnessLabel.setBuddy(robustnessComboBox)

        layout = QVBoxLayout()
        layout.addWidget(radioButton1)
        layout.addStretch(1)
        layout.addWidget(radioButton2)
        layout.addStretch(1)
        layout.addWidget(radioButton3)
        layout.addWidget(robustnessLabel)
        layout.addWidget(robustnessComboBox)
        layout.addStretch(1)
        layout.addWidget(radioButton4)
        self.sidebar.setLayout(layout)

    def createPanel(self):
        self.panel = QGroupBox("")

        titleLabel = QLabel("Title")
        titleFont = QFont()
        titleFont.setPointSize(24);
        titleLabel.setFont(titleFont)

        layout = QVBoxLayout()
        layout.addWidget(titleLabel)
        self.panel.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myapp = MainWindow()
    myapp.show()
    sys.exit(app.exec_())
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QGroupBox, QLabel, QVBoxLayout, QPushButton, QPlainTextEdit,
        QProgressBar)

from parameter_privacy import ParameterWidget

class PanelPrivacy(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, parent=parent)
        #self.setStyleSheet("background-color: #496e9c;")
        self.p = None  # Default empty value.
        titleLabel = QLabel("Privacy")
        # titleLabel.setStyleSheet("color: white;"
        #                     "background-color: #496e9c;")
        #                     #"border-style: solid;"
        #                     #"border-width: 1px;"
        #                     #"border-color: white;"
        #                     #"border-radius: 1px";
        titleFont = QFont()
        titleFont.setPointSize(24)
        titleLabel.setFont(titleFont)

        self.parameters = ParameterWidget()
        self.btn_submit = QPushButton("Check parameters")
        self.btn_submit.pressed.connect(self.submit_job)
        # self.btn_submit.setStyleSheet("color: white;"
        #                     "background-color: #496e9c;"
        #                     "border-style: solid;"
        #                     "border-width: 1px;"
        #                     "border-color: white;"
        #                     "border-radius: 1px")
        self.btn_run_experiment = QPushButton("Run experiment")
        self.btn_run_experiment.pressed.connect(self.run_experiment)
        # self.btn_run_experiment.setStyleSheet("color: white;"
        #                     "background-color: #496e9c;"
        #                     "border-style: solid;"
        #                     "border-width: 1px;"
        #                     "border-color: white;"
        #                     "border-radius: 1px")
        # self.btn_generate_dataset = QPushButton("Generate dataset")
        # self.btn_generate_dataset.pressed.connect(self.generate_dataset)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        # self.text.setStyleSheet("color: black;"
        #                     "background-color: #f68026;"
        #                     "border-style: solid;"
        #                     "border-width: 3px;"
        #                     "border-color: white;"
        #                     "border-radius: 3px")
        self.progress = QProgressBar()
        self.progress.setRange(0,100)

        # layout
        layout = QVBoxLayout(self)
        layout.addWidget(titleLabel)
        self.setLayout(layout)
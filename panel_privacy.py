from PyQt5.QtGui import QFont
from PyQt5.QtCore import QProcess
from PyQt5.QtWidgets import (QGridLayout, QGroupBox, QLabel, QVBoxLayout, QPushButton, QPlainTextEdit, QProgressBar, QWidget)

import os
import re

from parameter_privacy import ParameterWidget

class PanelPrivacy(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, parent=parent)
        self.p = None  # Default empty value.
        titleLabel = QLabel("Privacy")
        titleFont = QFont()
        titleFont.setPointSize(24)
        titleLabel.setFont(titleFont)

        self.parameters = ParameterWidget()
        self.btn_submit = QPushButton("Check parameters")
        self.btn_submit.pressed.connect(self.submit_job)
        self.btn_run_experiment = QPushButton("Run experiment")
        self.btn_run_experiment.pressed.connect(self.run_experiment)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.progress = QProgressBar()
        self.progress.setRange(0,100)

        # layout
        layout = QVBoxLayout(self)
        layout.addWidget(titleLabel)
        layout.addWidget(self.parameters)
        layout.addWidget(self.btn_submit)
        layout.addWidget(self.btn_run_experiment)
        #layout.addWidget(self.btn_generate_dataset)
        layout.addWidget(self.text)
        layout.addWidget(self.progress)

        bottomLayoutWrapper = QWidget()
        bottomLayoutWrapper.setObjectName("bottomLayoutWrapper")
        bottomLayoutWrapper.setLayout(layout)

        subLayout = QGridLayout()
        subLayout.addWidget(bottomLayoutWrapper)
    
        self.setLayout(subLayout)

    def generate_dataset(self):
        if self.p is None:  # No process running.
            self.message("Executing process")
            self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
            self.p.readyReadStandardOutput.connect(self.handle_stdout)
            self.p.readyReadStandardError.connect(self.handle_stderr)
            self.p.stateChanged.connect(self.handle_state)
            return_path = "../../../../"
            self.p.finished.connect(lambda: self.process_finished(return_path))  # Clean up once complete.
            os.chdir(" ")
            script = " "
            self.p.start(f"{return_path}venv/Scripts/python.exe", script.split(" "))

    # pip install scipy, torch, torchvision, six, progress, darts
    def run_experiment(self):
        if self.p is None:  # No process running.
            self.message("Executing process")
            self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
            self.p.readyReadStandardOutput.connect(self.handle_stdout)
            self.p.readyReadStandardError.connect(self.handle_stderr)
            self.p.stateChanged.connect(self.handle_state)
            return_path = "../../"
            self.p.finished.connect(lambda: self.process_finished(return_path))  # Clean up once complete.
            os.chdir("algorithms/invertingGradients")
            model = self.parameters.model
            dataset = self.parameters.dataset
            cost_fn = self.parameters.cost_fn
            indices = self.parameters.indices
            restarts = self.parameters.restarts
            target_id = self.parameters.target_id
            script = f"reconstruct_image.py --model {model} --dataset {dataset} --trained_model --cost_fn {cost_fn} --indices {indices} --restarts {restarts} --save_image --target_id {target_id}"
            self.p.start(f"{return_path}venv/Scripts/python.exe", script.split(" "))

    def message(self, s):
        self.text.appendPlainText(s)

    def submit_job(self):
        self.message(f"Model: {self.parameters.model} / Dataset: {self.parameters.dataset} / Cost_fn: {self.parameters.cost_fn} / Indices: {self.parameters.indices} / Restarts: {self.parameters.restarts} / Target_id: {self.parameters.target_id}")

    def handle_stderr(self):
        data = self.p.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        #Extract progress if it is in the data.
        progress = simple_percent_parser(stderr)
        if progress:
            self.progress.setValue(progress)
        self.message(stderr)
        
    def handle_stdout(self):
        data = self.p.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        self.message(stdout)

    def handle_state(self, state):
        states = {
            QProcess.NotRunning: 'Not running',
            QProcess.Starting: 'Starting',
            QProcess.Running: 'Running',
        }
        state_name = states[state]
        self.message(f"State changed: {state_name}")

    def process_finished(self, return_path=None):
        self.message("Process finished.")
        self.p = None
        if return_path is not None:
            os.chdir(return_path)

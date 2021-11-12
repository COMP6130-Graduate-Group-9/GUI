from PyQt5.QtGui import QFont
from PyQt5.QtCore import QProcess
from PyQt5.QtWidgets import (QGroupBox, QLabel, QVBoxLayout, QPushButton, QPlainTextEdit, QProgressBar)

import os
import re

# A regular expression, to extract the % complete.
progress_re = re.compile("Total complete: (\d+)%")

def simple_percent_parser(output):
    """
    Matches lines using the progress_re regex,
    returning a single integer for the % progress.
    """
    m = progress_re.search(output)
    if m:
        pc_complete = m.group(1)
        return int(pc_complete)

class PanelPrivacy(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, parent=parent)

        self.p = None  # Default empty value.

        titleLabel = QLabel("Privacy")
        titleFont = QFont()
        titleFont.setPointSize(24)
        titleLabel.setFont(titleFont)
        self.btn_generate_dataset = QPushButton("Generate dataset")
        self.btn_generate_dataset.pressed.connect(self.generate_dataset)
        self.btn_run_experiment = QPushButton("Run experiment")
        self.btn_run_experiment.pressed.connect(self.run_experiment)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)

        self.progress = QProgressBar()
        self.progress.setRange(0,100)

        layout = QVBoxLayout(self)
        layout.addWidget(titleLabel)
        layout.addWidget(self.btn_run_experiment)
        #layout.addWidget(self.btn_generate_dataset)
        layout.addWidget(self.text)
        layout.addWidget(self.progress)
        self.setLayout(layout)

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

    # pip install scipy, torch, torchvision, six, progress
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
            script = "reconstruct_image.py --model ResNet20-4 --dataset CIFAR10 --trained_model --cost_fn sim --indices def --restarts 32 --save_image --target_id -1"
            self.p.start(f"python3", script.split(" "))

    def message(self, s):
        self.text.appendPlainText(s)

    def handle_stderr(self):
        data = self.p.readAllStandardError()
        stderr = bytes(data).decode("utf8")
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

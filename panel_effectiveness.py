from PyQt5.QtGui import QFont
from PyQt5.QtCore import QProcess
from PyQt5.QtWidgets import (QGroupBox, QLabel, QVBoxLayout, QPushButton, QPlainTextEdit)

import os

class PanelEffectiveness(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, parent=parent)

        self.p = None  # Default empty value.

        titleLabel = QLabel("Effectiveness")
        titleFont = QFont()
        titleFont.setPointSize(24)
        titleLabel.setFont(titleFont)

        self.btn_generate_dataset = QPushButton("Generate dataset")
        self.btn_generate_dataset.pressed.connect(self.generate_dataset)
        self.btn_run_experiment = QPushButton("Run experiment")
        self.btn_run_experiment.pressed.connect(self.run_experiment)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(titleLabel)
        layout.addWidget(self.btn_generate_dataset)
        layout.addWidget(self.btn_run_experiment)
        layout.addWidget(self.text)
        self.setLayout(layout)

    def message(self, s):
        self.text.appendPlainText(s)

    def generate_dataset(self):
        if self.p is None:  # No process running.
            self.message("Executing process")
            self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
            self.p.readyReadStandardOutput.connect(self.handle_stdout)
            self.p.readyReadStandardError.connect(self.handle_stderr)
            self.p.stateChanged.connect(self.handle_state)
            return_path = "../../../../"
            self.p.finished.connect(lambda: self.process_finished(return_path))  # Clean up once complete.
            os.chdir("algorithms/FedGen/data/Mnist")
            script = "generate_niid_dirichlet.py --n_class 10 --sampling_ratio 0.5 --alpha 0.1 --n_user 20"
            self.p.start(f"{return_path}venv/Scripts/python.exe", script.split(" "))

    def run_experiment(self):
        if self.p is None:  # No process running.
            self.message("Executing process")
            self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
            self.p.readyReadStandardOutput.connect(self.handle_stdout)
            self.p.readyReadStandardError.connect(self.handle_stderr)
            self.p.stateChanged.connect(self.handle_state)
            return_path = "../../"
            self.p.finished.connect(lambda: self.process_finished(return_path))  # Clean up once complete.
            os.chdir("algorithms/FedGen")
            script = "main.py --dataset Mnist-alpha0.1-ratio0.5 --algorithm FedGen --batch_size 32 --num_glob_iters 200 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --personal_learning_rate 0.01 --times 3"
            self.p.start(f"{return_path}venv/Scripts/python.exe", script.split(" "))

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
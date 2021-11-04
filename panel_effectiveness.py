from PyQt5.QtGui import QFont
from PyQt5.QtCore import QProcess
from PyQt5.QtWidgets import (QGroupBox, QLabel, QVBoxLayout, QPushButton, QPlainTextEdit)

class PanelEffectiveness(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, parent=parent)

        self.p = None  # Default empty value.

        titleLabel = QLabel("Effectiveness")
        titleFont = QFont()
        titleFont.setPointSize(24)
        titleLabel.setFont(titleFont)

        self.btn = QPushButton("Generate dataset")
        self.btn.pressed.connect(self.generate_dataset)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(titleLabel)
        layout.addWidget(self.btn)
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
            self.p.finished.connect(self.process_finished)  # Clean up once complete.
            script = "algorithms/FedGen/data/Mnist/generate_niid_dirichlet.py --n_class 10 --sampling_ratio 0.5 --alpha 0.1 --n_user 20"
            self.p.start("venv/Scripts/python.exe", script.split(" "))

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
    
    def process_finished(self):
        self.message("Process finished.")
        self.p = None
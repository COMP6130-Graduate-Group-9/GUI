from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QProgressBar, QPlainTextEdit)

class GeneralJob(QWidget):
    def __init__(self) -> None:
        super().__init__()

        general_job_layout = QVBoxLayout()
        self.setLayout(general_job_layout)
        
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFormat('This is progress bar')

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)

        general_job_layout.addWidget(self.progress)
        general_job_layout.addWidget(self.text)
    
    @pyqtSlot(str)
    def update_status(self, status):
        self.text.appendPlainText(status)
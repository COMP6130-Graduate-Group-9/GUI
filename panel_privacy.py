from PyQt5.QtGui import QFont
from PyQt5.QtCore import QProcess
from PyQt5.QtWidgets import (QGridLayout, QGroupBox, QLabel, QVBoxLayout, QPushButton, 
    QPlainTextEdit, QProgressBar, QWidget, QStackedLayout)

from privacy import job_widget

class PanelPrivacy(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, parent=parent)
        self.p = None  # Default empty value.
        titleLabel = QLabel("Privacy")
        titleFont = QFont()
        titleFont.setPointSize(20)
        titleLabel.setFont(titleFont)

        self.current_job_display = QStackedLayout()
        # parameters display
        self.parameters_container = job_widget.Parameters(self)
        # main job display
        self.general_job_container = job_widget.GeneralJob(self)
        # results display
        self.results_container = job_widget.Results(self)
        
        self.current_job_display.addWidget(self.parameters_container)
        self.current_job_display.addWidget(self.general_job_container)
        self.current_job_display.addWidget(self.results_container)
        self.current_job_display.setCurrentIndex(0)

        # layout
        layout = QVBoxLayout(self)
        layout.addWidget(titleLabel)
        layout.addLayout(self.current_job_display)

        bottomLayoutWrapper = QWidget()
        bottomLayoutWrapper.setObjectName("bottomLayoutWrapper")
        bottomLayoutWrapper.setLayout(layout)

        subLayout = QGridLayout()
        subLayout.addWidget(bottomLayoutWrapper)
    
        self.setLayout(subLayout)

    def switch_current_job_display(self, index):
        self.current_job_display.setCurrentIndex(index)
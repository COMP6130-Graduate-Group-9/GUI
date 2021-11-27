from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel)

class Timer(QWidget):
    def __init__(self):
        super().__init__()

        self.counter = 0
        self.elapsed_time = None

        layout = QVBoxLayout(self)

        elapsed_time_label = QLabel()
        elapsed_time_label.setText("Elapsed time")
        self.time_display = QLabel()
        self.time_display.setText("00:00:00")
        timerFont = QFont()
        timerFont.setPointSize(16)
        self.time_display.setFont(timerFont)
        layout.addWidget(elapsed_time_label)
        layout.addWidget(self.time_display)

        self.timer = QTimer()
        self.timer.timeout.connect(self.display)

    def start(self):
        self.timer.start(1000)

    def stop(self):
        self.timer.stop()

    def reset(self):
        self.counter = 0
        self.elapsed_time = None
        self.time_display.setText("00:00:00")
    
    def display(self):
        self.counter += 1
        second = self.counter % 60
        minute = (self.counter // 60) % 60
        hour = self.counter // 3600
        self.elapsed_time = f"{hour:02}:{minute:02}:{second:02}"
        self.time_display.setText(self.elapsed_time)
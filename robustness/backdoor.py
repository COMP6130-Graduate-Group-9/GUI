from PyQt5.QtCore import pyqtSlot, QProcess, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QStackedLayout, QWidget, QGridLayout, QLabel, QHBoxLayout,
    QComboBox, QSpinBox, QPushButton, QVBoxLayout, QProgressBar,
    QPlainTextEdit, QDoubleSpinBox)

import yaml, re, os
from process_manager import ProcessManager

class Container(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.stackedLayout = QStackedLayout()
        self.parameters = Parameters(self)
        self.general_job = GeneralJob(self)
        self.results = Results(self)
        self.stackedLayout.addWidget(self.parameters)
        self.stackedLayout.addWidget(self.general_job)
        self.stackedLayout.addWidget(self.results)
        self.stackedLayout.setCurrentIndex(0)

        layout.addLayout(self.stackedLayout)

    def switch_current_job_display(self, index):
        self.stackedLayout.setCurrentIndex(index)


class Parameters(QWidget):
    def __init__(self, main_panel):
        super().__init__()

        self.main_panel = main_panel

        self.task = "MNIST"
        self.learning_rate = 0.01
        self.epochs = 5

        task_list = [
            "MNIST"
        ]

        grid = QGridLayout()  
        self.setLayout(grid)

        description = QLabel("Please specify your parameters and submit the job")

        task_box = QHBoxLayout()
        task_input = QComboBox()
        for task in task_list:
            task_input.addItem(task, task)
        task_input.currentTextChanged.connect(self.task_text_changed)
        curr_task_idx = task_input.findData(self.task)
        task_input.setCurrentIndex(curr_task_idx)
        task_label = QLabel("Task:")
        task_label.setBuddy(task_input)
        task_box.addWidget(task_label)
        task_box.addWidget(task_input)

        learning_rate_box = QHBoxLayout()
        learning_rate_input = QDoubleSpinBox()
        learning_rate_input.setValue(self.learning_rate)
        learning_rate_input.valueChanged.connect(lambda i: self.value_changed(i, "learning_rate"))
        learning_rate_label = QLabel("Learning rate:")
        learning_rate_label.setBuddy(learning_rate_input)
        learning_rate_box.addWidget(learning_rate_label)
        learning_rate_box.addWidget(learning_rate_input)

        epochs_box = QHBoxLayout()
        epochs_input = QSpinBox()
        epochs_input.setValue(self.epochs)
        epochs_input.valueChanged.connect(lambda i: self.value_changed(i, "epochs"))
        epochs_label = QLabel("Epochs:")
        epochs_label.setBuddy(epochs_input)
        epochs_box.addWidget(epochs_label)
        epochs_box.addWidget(epochs_input)

        btn_submit = QPushButton("SUBMIT")
        btn_submit.pressed.connect(self.submit_job)

        grid.addWidget(description, 0, 0, 1, 3)
        grid.addLayout(task_box, 1, 0, 2, 1)
        grid.addLayout(learning_rate_box, 1, 1, 2, 1)
        grid.addLayout(epochs_box, 1, 2, 2, 1)
        grid.addWidget(btn_submit, 2, 0, 1, 3)

    def value_changed(self, i, name):
        self.__dict__[name] = i

    def task_text_changed(self, task):
        self.task = task

    def submit_job(self):
        with open('robustness/params.yaml') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            params['task'] = self.task
            params['lr'] = self.learning_rate
            params['epochs'] = self.epochs
        with open('robustness/params.yaml', 'w') as f:
            yaml.dump(params, f)

        self.main_panel.switch_current_job_display(1)

        exec_dir = f"algorithms/backdoors101"
        return_path = "../../"
        self.manager = ProcessManager(exec_dir, return_path)
        self.manager.started.connect(self.main_panel.general_job.initialize)
        self.manager.finished.connect(self.main_panel.general_job.complete)
        self.manager.textChanged.connect(self.main_panel.general_job.update_status)

        params_path = os.path.join(os.getcwd(), 'robustness/params.yaml')
        script = f"training.py --name {self.task} --params {params_path} --commit none"
        self.manager.run_script(script)

class GeneralJob(QWidget):
    def __init__(self, main_panel):
        super().__init__()

        self.main_panel = main_panel

        layout = QGridLayout()
        self.setLayout(layout)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)

        right_box = QVBoxLayout()
        self.timer = Timer()
        right_box.addWidget(self.timer)
        self.btn_view_results = QPushButton("View results")
        self.btn_view_results.pressed.connect(self.view_results)
        self.btn_view_results.setEnabled(False)
        right_box.addWidget(self.btn_view_results)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.pressed.connect(self.cancel)
        right_box.addWidget(self.btn_cancel)

        layout.addWidget(self.progress, 0, 0, 1, 5)
        layout.addWidget(self.text, 1, 0, 1, 1)
        layout.addLayout(right_box, 1, 1, 1, 1)

    @pyqtSlot()
    def initialize(self):
        self.progress.setValue(0)
        self.timer.start()
    
    @pyqtSlot(str)
    def update_status(self, status):
        print(status)
        self.track_global_result(status)
        self.text.appendPlainText(status)

    @pyqtSlot(int, QProcess.ExitStatus)
    def complete(self, exitCode, exitStatus):
        if exitStatus == 0:
            self.timer.stop()
            self.progress.setValue(100)
            self.btn_cancel.setParent(None)
            self.btn_view_results.setEnabled(True)

            self.main_panel.results_container.elapsed_time = self.timer.elapsed_time
            self.main_panel.results_container.rec_loss = self.rec_loss
            self.main_panel.results_container.ground_truth_filename = self.ground_truth_filename
            self.main_panel.results_container.output_filename = self.output_filename
            self.main_panel.results_container.populate_content()

    def view_results(self):
        pass

    def cancel(self):
        pass

    def reset(self):
        pass

    def track_global_result(self, status):
        re_iteration = re.findall(r'It: (\d+)\.', status)
        re_rec_loss = re.findall(r'Rec. loss: ([0-9]*[.]?[0-9]+)', status)
        if len(re_iteration) > 0:
            percentage = int(re_iteration[0]) / 3_000 * 100
            self.progress.setValue(percentage)
        if len(re_rec_loss) > 0:
            self.rec_loss = re_rec_loss[0]

class Results(QWidget):
    def __init__(self, main_panel):
        super().__init__()

        self.main_panel = main_panel

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
from PyQt5.QtCore import pyqtSlot, QProcess, QPoint, QRect
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (QStackedLayout, QWidget, QGridLayout, QLabel, QHBoxLayout,
    QComboBox, QSpinBox, QPushButton, QVBoxLayout, QProgressBar,
    QPlainTextEdit, QDoubleSpinBox, QTableWidget, QSizePolicy,
    QTableWidgetItem, QToolTip)

import yaml, re, os, time
from util.process_manager import ProcessManager
from util.timer import Timer

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
        self.backdoor_label = None

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
            self.backdoor_label = params['backdoor_label']
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

            self.main_panel.results.elapsed_time = self.timer.elapsed_time
            self.main_panel.results.populate_content()

    def view_results(self):
        self.main_panel.switch_current_job_display(2)

    def cancel(self):
        if self.main_panel.parameters.manager:
            self.main_panel.parameters.manager.stop()
        self.reset()

        self.main_panel.switch_current_job_display(0)

    def reset(self):
        self.timer.stop()
        self.timer.reset()
        self.text.clear()
        self.progress.setValue(0)
        self.btn_cancel.setParent(self)
        self.btn_view_results.setEnabled(False)

    def track_global_result(self, status):
        re_iteration = re.findall(r'Epoch:\s+(\d+)\.', status)
        epoch = self.main_panel.parameters.epochs
        if len(re_iteration) > 0:
            percentage = int(re_iteration[0]) - 1 / epoch * 100
            print(epoch, percentage)
            self.progress.setValue(percentage)

class Results(QWidget):
    def __init__(self, main_panel):
        super().__init__()

        self.main_panel = main_panel

        grid = QGridLayout()
        self.setLayout(grid)

        sample1_box = QHBoxLayout()
        self.sample1_image = QLabel()
        self.sample1_label = QLabel()
        sample1_box.addWidget(self.sample1_image)
        sample1_box.addWidget(self.sample1_label)

        sample2_box = QHBoxLayout()
        self.sample2_image = QLabel()
        self.sample2_label = QLabel()
        sample2_box.addWidget(self.sample2_image)
        sample2_box.addWidget(self.sample2_label)

        sample3_box = QHBoxLayout()
        self.sample3_image = QLabel()
        self.sample3_label = QLabel()
        sample3_box.addWidget(self.sample3_image)
        sample3_box.addWidget(self.sample3_label)

        self.parameters = {
            'Dataset': main_panel.parameters.task,
            'Learning rate': main_panel.parameters.learning_rate,
            'Total epochs': main_panel.parameters.epochs
        }
        
        self.table = QTableWidget(self)
        self.table.setColumnCount(2)
        self.table.setRowCount(4)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.table.setSizePolicy(sizePolicy)
        
        for idx, val in enumerate(self.parameters.items()):
            self.table.setItem(idx, 0, QTableWidgetItem(val[0]))
            self.table.setItem(idx, 1, QTableWidgetItem(str(val[1])))

        self.btn_save_results = QPushButton("Save results")
        self.btn_save_results.clicked.connect(self.save_results)
        btn_restart = QPushButton("Restart")
        btn_restart.clicked.connect(self.return_to_parameters)

        grid.addLayout(sample1_box, 0, 0, 1, 1)
        grid.addLayout(sample2_box, 1, 0, 1, 1)
        grid.addLayout(sample3_box, 2, 0, 1, 1)
        grid.addWidget(self.table, 0, 1, 3, 1)
        grid.addWidget(self.btn_save_results, 3, 0, 1, 1)
        grid.addWidget(btn_restart, 3, 1, 1, 1)
        

    def populate_content(self):
        sample1_pixmap = QPixmap(f"algorithms/backdoors101/images/sample-1.png")
        self.sample1_image.setPixmap(sample1_pixmap)
        sample2_pixmap = QPixmap(f"algorithms/backdoors101/images/sample-2.png")
        self.sample2_image.setPixmap(sample2_pixmap)
        sample3_pixmap = QPixmap(f"algorithms/backdoors101/images/sample-3.png")
        self.sample3_image.setPixmap(sample3_pixmap)
        self.sample1_label.setText(f"Backdoor label: {self.main_panel.parameters.backdoor_label}")
        self.sample2_label.setText(f"Backdoor label: {self.main_panel.parameters.backdoor_label}")
        self.sample3_label.setText(f"Backdoor label: {self.main_panel.parameters.backdoor_label}")

        self.table.setItem(3, 0, QTableWidgetItem('Elapsed time'))
        self.table.setItem(3, 1, QTableWidgetItem(self.elapsed_time))
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        # self.reconstruct_loss_display.setText(f"Reconstruct loss: {self.rec_loss}")

    def save_results(self):
        logs_path = os.path.join(os.getcwd(), 'logs')
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)

        lines = [
            'Result',
            # f'Accuracy: {self.accuracy}',
            # f'Loss: {self.loss}',
            f'Elapsed time: {self.elapsed_time}',
            '',
            'Parameters'
        ]
        lines += [f'{p[0]}: {p[1]}' for p in self.parameters.items()]
        t = time.localtime()
        current_time = time.strftime("%m-%d-%Y_%H-%M-%S", t)
        filename = f'robustness-backdoor-attack-{current_time}.txt'
        file_path = os.path.join(logs_path, filename)
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))

        QToolTip.showText(self.btn_save_results.mapToGlobal(QPoint(0,0)), f'File saved to {file_path}', self.btn_save_results, QRect(), 1000)
    
    def return_to_parameters(self):
        if self.main_panel.parameters.manager:
            self.main_panel.parameters.manager.stop()
        self.main_panel.general_job.reset()

        self.main_panel.switch_current_job_display(0)
import os
import re
import time
from PyQt5.QtCore import QPoint, QProcess, QRect, pyqtSlot
from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QSpinBox, QStackedLayout, QTableWidget, QTableWidgetItem, QToolTip, QVBoxLayout, QWidget

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

        self.dataset = "CIFAR10"
        self.benign_known = "unknown"
        self.aggregation = "bulyan"
        self.attack = "fang"
        self.epoch = 100
        self.learning_rate = 0.5

        grid = QGridLayout()
        self.setLayout(grid)

        description = QLabel("Please specify your parameters and submit the job")
        description.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        aggregation_list = [
            "Bulyan",
            "Multi-krum",
            "Median",
            "Trimmed-mean"
        ]

        attack_list = [
            "Fang",
            "Lie",
            "AGR-tailored",
            "Min-Max",
            "Min-sum"
        ]

        dataset_box = QHBoxLayout()
        dataset_input = QComboBox()
        dataset_input.addItem("CIFAR10", "CIFAR10")
        # dataset_input.currentTextChanged.connect(self.dataset_text_changed)
        # curr_dataset_idx = dataset_input.findData(self.dataset)
        # dataset_input.setCurrentIndex(curr_dataset_idx)
        dataset_label = QLabel("Dataset:")
        dataset_label.setBuddy(dataset_input)
        dataset_box.addWidget(dataset_label)
        dataset_box.addWidget(dataset_input)

        aggregation_box = QHBoxLayout()
        aggregation_input = QComboBox()
        for aggregation in aggregation_list:
            aggregation_input.addItem(aggregation, aggregation)
        aggregation_input.currentTextChanged.connect(self.aggregation_text_changed)
        aggregation_input.setCurrentText(aggregation_list[0])
        aggregation_label = QLabel("Aggregation:")
        aggregation_label.setBuddy(aggregation_input)
        aggregation_box.addWidget(aggregation_label)
        aggregation_box.addWidget(aggregation_input)

        attack_box = QHBoxLayout()
        attack_input = QComboBox()
        for attack in attack_list:
            attack_input.addItem(attack, attack)
        attack_input.currentTextChanged.connect(self.attack_text_changed)
        attack_input.setCurrentText(attack_list[0])
        attack_label = QLabel("Attack:")
        attack_label.setBuddy(attack_input)
        attack_box.addWidget(attack_label)
        attack_box.addWidget(attack_input)

        benign_box = QHBoxLayout()
        benign_input = QComboBox()
        benign_input.addItem("Unknown", "Unknown")
        benign_input.addItem("Known", "known")
        benign_input.currentTextChanged.connect(self.benign_text_changed)
        benign_input.setCurrentText("Unknown")
        benign_label = QLabel("Benign gradient:")
        benign_label.setBuddy(benign_input)
        benign_box.addWidget(benign_label)
        benign_box.addWidget(benign_input)

        epoch_box = QHBoxLayout()
        epoch_input = QSpinBox()
        epoch_input.setMaximum(100000)
        epoch_input.setValue(self.epoch)
        epoch_input.valueChanged.connect(lambda i: self.value_changed(i, "epoch"))
        epoch_label = QLabel("Epoch:")
        epoch_label.setBuddy(epoch_input)
        epoch_box.addWidget(epoch_label)
        epoch_box.addWidget(epoch_input)

        learning_rate_box = QHBoxLayout()
        learning_rate_input = QDoubleSpinBox()
        learning_rate_input.setValue(self.learning_rate)
        learning_rate_input.valueChanged.connect(lambda i: self.value_changed(i, "learning_rate"))
        learning_rate_label = QLabel("Learning rate:")
        learning_rate_label.setBuddy(learning_rate_input)
        learning_rate_box.addWidget(learning_rate_label)
        learning_rate_box.addWidget(learning_rate_input)

        btn_submit = QPushButton("SUBMIT")
        btn_submit.pressed.connect(self.submit_job)

        grid.addWidget(description, 0, 0, 1, 3)
        grid.addLayout(dataset_box, 1, 0, 1, 1)
        grid.addLayout(aggregation_box, 1, 1, 1, 1)
        grid.addLayout(attack_box, 1, 2, 1, 1)
        grid.addLayout(benign_box, 2, 0, 1, 1)
        grid.addLayout(epoch_box, 2, 1, 1, 1)
        grid.addLayout(learning_rate_box, 2, 2, 1, 1)
        grid.addWidget(btn_submit, 3, 0, 1, 3)

    def value_changed(self, i, name):
        self.__dict__[name] = i

    def aggregation_text_changed(self, aggregation):
        if aggregation == "Bulyan":
            self.aggregation = "bulyan"
        elif aggregation == "Multi-krum":
            self.aggregation = "multi-krum"
        elif aggregation == "Median":
            self.aggregation = "median"
        elif aggregation == "Trimmed-mean":
            self.aggregation = "trmean"

    def attack_text_changed(self, text):
        if text == "Fang":
            self.attack = "fang"
        elif text == "Lie":
            self.attack = "lie"
        elif text == "AGR-tailored attack":
            self.attack = "agr-tailored"
        elif text == "Min-Max":
            self.attack = "min-max"
        elif text == "Min-sum":
            self.attack = "min-sum"

    def benign_text_changed(self, text):
        if text == "Known":
            self.benign_known = "known"
        else:
            self.benign_known = "unknown"

    def submit_job(self):
        self.main_panel.switch_current_job_display(1)

        exec_dir = f"algorithms/NDSS21-Model-Poisoning/cifar10"
        return_path = "../../../"
        self.manager = ProcessManager(exec_dir, return_path)
        self.manager.started.connect(self.main_panel.general_job.initialize)
        self.manager.finished.connect(self.main_panel.general_job.complete)
        self.manager.textChanged.connect(self.main_panel.general_job.update_status)

        benign_known = self.benign_known
        aggregation = self.aggregation
        attack = self.attack
        epoch = self.epoch
        learning_rate = self.learning_rate
        script = f"portal.py --benign_known {benign_known} --aggregation {aggregation} --attack {attack} --epoch {epoch} --learning_rate {learning_rate}"
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

            self.main_panel.results.get_latest_elapsed_time()
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
        re_iteration = re.findall(r'\s+e\s+(\d+)\s+', status)
        re_val_loss = re.findall(r'val loss ([0-9]*[.]?[0-9]+)', status)
        re_val_acc = re.findall(r'best val_acc ([0-9]*[.]?[0-9]+)', status)
        re_test_acc = re.findall(r'te_acc ([0-9]*[.]?[0-9]+)', status)
        if len(re_iteration) > 0:
            percentage = int(re_iteration[0]) / self.main_panel.parameters.epoch * 100
            self.progress.setValue(percentage)
        if len(re_val_loss) > 0:
            self.main_panel.results.val_loss = re_val_loss[0]
        if len(re_val_acc) > 0:
            self.main_panel.results.val_acc = re_val_acc[0]
        if len(re_test_acc) > 0:
            self.main_panel.results.test_acc = re_test_acc[0]
        

class Results(QWidget):
    def __init__(self, main_panel):
        super().__init__()

        self.main_panel = main_panel

        self.val_loss = None
        self.val_acc = None
        self.test_acc = None

        layout = QGridLayout()
        self.setLayout(layout)

        numerical_result_box = QVBoxLayout()
        self.val_loss_display = QLabel()
        numerical_result_box.addWidget(self.val_loss_display)
        self.val_acc_display = QLabel()
        numerical_result_box.addWidget(self.val_acc_display)
        self.test_acc_display = QLabel()
        numerical_result_box.addWidget(self.test_acc_display)
        self.elapsed_time_display = QLabel()
        numerical_result_box.addWidget(self.elapsed_time_display)

        table = QTableWidget(self)
        table.setColumnCount(2)
        table.setRowCount(6)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        table.setSizePolicy(sizePolicy)

        self.parameters = {
            'Dataset': "CIFAR10",
            'Gradient of benign client known': self.main_panel.parameters.benign_known,
            'Aggregation model': self.main_panel.parameters.aggregation,
            'Attack': self.main_panel.parameters.attack,
            'Epoch': self.main_panel.parameters.epoch,
            'Learning rate': self.main_panel.parameters.learning_rate,
        }

        for idx, val in enumerate(self.parameters.items()):
            table.setItem(idx, 0, QTableWidgetItem(val[0]))
            table.setItem(idx, 1, QTableWidgetItem(str(val[1])))

        table.resizeColumnsToContents()
        table.resizeRowsToContents()

        self.btn_save_results = QPushButton("Save results")
        self.btn_save_results.clicked.connect(self.save_results)
        btn_restart = QPushButton("Restart")
        btn_restart.clicked.connect(self.return_to_parameters)

        layout.addLayout(numerical_result_box, 0, 0, 1, 1)
        layout.addWidget(table, 0, 1, 1, 1)
        layout.addWidget(self.btn_save_results, 1, 0, 1, 1)
        layout.addWidget(btn_restart, 1, 1, 1, 1)

    def populate_content(self):
        if self.val_loss:
            self.val_loss_display.setText(f"Validation loss: {float(self.val_loss):.4f}")
        if self.val_acc:
            self.val_acc_display.setText(f"Validation accuracy: {float(self.val_acc):.4f}")
        if self.test_acc:
            self.test_acc_display.setText(f"Test accuracy: {float(self.test_acc):.4f}")

    def get_latest_elapsed_time(self):
        self.elapsed_time = self.main_panel.general_job.timer.elapsed_time
        self.elapsed_time_display.setText(f"Elapsed time: {self.elapsed_time}")

    def save_results(self):
        logs_path = os.path.join(os.getcwd(), 'logs')
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)

        lines = [
            'Result',
            f'Validation loss: {self.val_loss}',
            f'Validation accuracy: {self.val_acc}',
            f'Test accuracy: {self.test_acc}',
            f'Elapsed time: {self.elapsed_time}',
            '',
            'Parameters'
        ]
        lines += [f'{p[0]}: {p[1]}' for p in self.parameters.items()]
        t = time.localtime()
        current_time = time.strftime("%m-%d-%Y_%H-%M-%S", t)
        filename = f'robustness-model-poisoning-{self.main_panel.parameters.attack}-{self.main_panel.parameters.aggregation}-{current_time}.txt'
        file_path = os.path.join(logs_path, filename)
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))

        QToolTip.showText(self.btn_save_results.mapToGlobal(QPoint(0,0)), f'File saved to {file_path}', self.btn_save_results, QRect(), 1000)
    
    def return_to_parameters(self):
        if self.main_panel.parameters.manager:
            self.main_panel.parameters.manager.stop()
        self.main_panel.general_job.reset()

        self.main_panel.switch_current_job_display(0)
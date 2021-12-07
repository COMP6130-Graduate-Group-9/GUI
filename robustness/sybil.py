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

        self.dataset = "MNIST"
        self.iterations = 1000
        self.source = "1"
        self.target = "7"
        self.sybil = 5

        grid = QGridLayout()
        self.setLayout(grid)

        description = QLabel("Please specify your parameters and submit the job")
        description.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        label_list = [str(i) for i in range(10)]

        dataset_box = QHBoxLayout()
        dataset_input = QComboBox()
        dataset_input.addItem("MNIST", "MNIST")
        dataset_label = QLabel("Dataset:")
        dataset_label.setBuddy(dataset_input)
        dataset_box.addWidget(dataset_label)
        dataset_box.addWidget(dataset_input)

        source_box = QHBoxLayout()
        source_input = QComboBox()
        for source in label_list:
            source_input.addItem(source, source)
        source_input.currentTextChanged.connect(lambda i: self.text_changed(i, "source"))
        curr_source_idx = source_input.findData(self.source)
        source_input.setCurrentIndex(curr_source_idx)
        source_label = QLabel("Source:")
        source_label.setBuddy(source_input)
        source_box.addWidget(source_label)
        source_box.addWidget(source_input)

        target_box = QHBoxLayout()
        target_input = QComboBox()
        for target in label_list:
            target_input.addItem(target, target)
        target_input.currentTextChanged.connect(lambda i: self.text_changed(i, "target"))
        curr_target_idx = target_input.findData(self.target)
        target_input.setCurrentIndex(curr_target_idx)
        target_label = QLabel("Source:")
        target_label.setBuddy(target_input)
        target_box.addWidget(target_label)
        target_box.addWidget(target_input)

        sybil_box = QHBoxLayout()
        sybil_input = QSpinBox()
        sybil_input.setMaximum(100000)
        sybil_input.setValue(self.sybil)
        sybil_input.valueChanged.connect(lambda i: self.value_changed(i, "sybil"))
        sybil_label = QLabel("Epoch:")
        sybil_label.setBuddy(sybil_input)
        sybil_box.addWidget(sybil_label)
        sybil_box.addWidget(sybil_input)

        btn_submit = QPushButton("SUBMIT")
        btn_submit.pressed.connect(self.submit_job)

        grid.addWidget(description, 0, 0, 1, 4)
        grid.addLayout(dataset_box, 1, 0, 1, 1)
        grid.addLayout(source_box, 1, 1, 1, 1)
        grid.addLayout(target_box, 1, 2, 1, 1)
        grid.addLayout(sybil_box, 1, 3, 1, 1)
        grid.addWidget(btn_submit, 2, 0, 1, 4)

    def value_changed(self, i, name):
        self.__dict__[name] = i

    def text_changed(self, text, name):
        setattr(self, name, text)
        
    def submit_job(self):
        self.main_panel.switch_current_job_display(1)

        exec_dir = f"algorithms/FoolsGold/ML/data/mnist"
        return_path = "../../../../../"
        self.manager = ProcessManager(exec_dir, return_path)
        self.manager.started.connect(self.main_panel.general_job.parse_mnist_initialize)
        self.manager.finished.connect(self.main_panel.general_job.parse_mnist_complete)
        self.manager.textChanged.connect(self.main_panel.general_job.parse_mnist_update_status)

        script = f"parse_mnist.py"
        self.manager.run_script(script)

class GeneralJob(QWidget):
    def __init__(self, main_panel):
        super().__init__()

        self.main_panel = main_panel
        self.manager = None
        self.curr_iteration = 0

        layout = QGridLayout()
        self.setLayout(layout)

        progress_box = QHBoxLayout()
        self.current_action = QLabel()
        self.current_action.setText('Parsing dataset.....')
        progress_box.addWidget(self.current_action)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        progress_box.addWidget(self.progress)

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

        layout.addLayout(progress_box, 0, 0, 1, 5)
        layout.addWidget(self.text, 1, 0, 1, 1)
        layout.addLayout(right_box, 1, 1, 1, 1)

    @pyqtSlot()
    def parse_mnist_initialize(self):
        self.main_panel.results.create_table()
    
    @pyqtSlot(str)
    def parse_mnist_update_status(self, status):
        print(status)
        self.text.appendPlainText(status)

    @pyqtSlot(int, QProcess.ExitStatus)
    def parse_mnist_complete(self, exitCode, exitStatus):
        if exitStatus == 0:
            self.mislabel_dataset()
    
    @pyqtSlot()
    def mislabel_initialize(self):
        self.current_action.setText('Creating mislabelled dataset.....')

    @pyqtSlot(str)
    def mislabel_update_status(self, status):
        print(status)
        self.text.appendPlainText(status)

    @pyqtSlot(int, QProcess.ExitStatus)
    def mislabel_complete(self, exitCode, exitStatus):
        if exitStatus == 0:
            self.run_experiment()

    @pyqtSlot()
    def initialize(self):
        self.progress.setValue(0)
        self.timer.start()
        self.current_action.setText('Running experiment.....')
    
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

    def mislabel_dataset(self):
        exec_dir = f"algorithms/FoolsGold/ML"
        return_path = "../../../"
        self.manager = ProcessManager(exec_dir, return_path)
        self.manager.started.connect(self.main_panel.general_job.mislabel_initialize)
        self.manager.finished.connect(self.main_panel.general_job.mislabel_complete)
        self.manager.textChanged.connect(self.main_panel.general_job.mislabel_update_status)

        source = self.main_panel.parameters.source
        target = self.main_panel.parameters.target
        script = f"code/misslabel_dataset.py mnist {source} {target}"
        self.manager.run_script(script)

    def run_experiment(self):
        exec_dir = f"algorithms/FoolsGold/ML"
        return_path = "../../../"
        self.manager = ProcessManager(exec_dir, return_path)
        self.manager.started.connect(self.main_panel.general_job.initialize)
        self.manager.finished.connect(self.main_panel.general_job.complete)
        self.manager.textChanged.connect(self.main_panel.general_job.update_status)

        source = self.main_panel.parameters.source
        target = self.main_panel.parameters.target
        sybil = self.main_panel.parameters.sybil
        iterations = self.main_panel.parameters.iterations
        script = f"code/ML_main.py mnist {iterations} {sybil}_{source}_{target}"
        self.manager.run_script(script)
    
    def view_results(self):
        self.main_panel.switch_current_job_display(2)

    def cancel(self):
        if self.main_panel.parameters.manager:
            self.main_panel.parameters.manager.stop()
        if self.manager:
            self.manager.stop()
        self.reset()

        self.main_panel.switch_current_job_display(0)

    def reset(self):
        self.timer.stop()
        self.timer.reset()
        self.text.clear()
        self.current_action.setText('Parsing dataset.....')
        self.progress.setValue(0)
        self.btn_cancel.setParent(self)
        self.btn_view_results.setEnabled(False)

    def track_global_result(self, status):
        re_validation = re.findall(r'Validation error: ([0-9]*[.]?[0-9]+)', status)
        re_train_error = re.findall(r'Train error: ([0-9]*[.]?[0-9]+)', status)
        re_test_error = re.findall(r'Test error: ([0-9]*[.]?[0-9]+)', status)
        re_acc_overall = re.findall(r'Accuracy overall: ([0-9]*[.]?[0-9]+)', status)
        if len(re_validation) > 0:
            self.curr_iteration += 100
            percentage = self.curr_iteration / self.main_panel.parameters.iterations * 100
            self.progress.setValue(percentage)
        if len(re_train_error) > 0:
            self.main_panel.results.record_value(float(re_train_error[0]), "train_error")
        if len(re_test_error) > 0:
            self.main_panel.results.record_value(float(re_test_error[0]), "test_error")
        if len(re_acc_overall) > 0:
            self.main_panel.results.record_value(float(re_acc_overall[0]), "accuracy_overall")
        
        

class Results(QWidget):
    def __init__(self, main_panel):
        super().__init__()

        self.main_panel = main_panel
        self.train_error = None
        self.test_error = None
        self.accuracy_overall = None

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        numerical_result_box = QVBoxLayout()
        self.train_error_display = QLabel()
        numerical_result_box.addWidget(self.train_error_display)
        self.test_error_display = QLabel()
        numerical_result_box.addWidget(self.test_error_display)
        self.accuracy_overall_display = QLabel()
        numerical_result_box.addWidget(self.accuracy_overall_display)
        self.elapsed_time_display = QLabel()
        numerical_result_box.addWidget(self.elapsed_time_display)

        self.btn_save_results = QPushButton("Save results")
        self.btn_save_results.clicked.connect(self.save_results)
        btn_restart = QPushButton("Restart")
        btn_restart.clicked.connect(self.return_to_parameters)

        self.layout.addLayout(numerical_result_box, 0, 0, 1, 1)
        self.layout.addWidget(self.btn_save_results, 1, 0, 1, 1)
        self.layout.addWidget(btn_restart, 1, 1, 1, 1)

    def create_table(self):
        table = QTableWidget(self)
        table.setColumnCount(2)
        table.setRowCount(5)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        table.setSizePolicy(sizePolicy)

        self.parameters = {
            'Dataset': "MNIST",
            'Iterations': self.main_panel.parameters.iterations,
            'Source label': self.main_panel.parameters.source,
            'Target label': self.main_panel.parameters.target,
            'Number of sybil': self.main_panel.parameters.sybil,
        }

        for idx, val in enumerate(self.parameters.items()):
            table.setItem(idx, 0, QTableWidgetItem(val[0]))
            table.setItem(idx, 1, QTableWidgetItem(str(val[1])))

        table.resizeColumnsToContents()
        table.resizeRowsToContents()
    
        self.layout.addWidget(table, 0, 1, 1, 1)


    def populate_content(self):
        if self.train_error:
            self.train_error_display.setText(f"Train error: {float(self.train_error)}")
        if self.test_error:
            self.test_error_display.setText(f"Test error: {float(self.test_error)}")
        if self.accuracy_overall:
            self.accuracy_overall_display.setText(f"Accuracy overall: {float(self.accuracy_overall)}")

    def get_latest_elapsed_time(self):
        self.elapsed_time = self.main_panel.general_job.timer.elapsed_time
        self.elapsed_time_display.setText(f"Elapsed time: {self.elapsed_time}")

    def record_value(self, value, name):
        setattr(self, name, value)

    def save_results(self):
        logs_path = os.path.join(os.getcwd(), 'logs')
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)

        lines = [
            'Result',
            f'Train error: {self.train_error}',
            f'Test error: {self.test_error}',
            f'Accuracy overall: {self.accuracy_overall}',
            f'Elapsed time: {self.elapsed_time}',
            '',
            'Parameters'
        ]
        lines += [f'{p[0]}: {p[1]}' for p in self.parameters.items()]
        t = time.localtime()
        current_time = time.strftime("%m-%d-%Y_%H-%M-%S", t)
        filename = f'robustness-sybil-{current_time}.txt'
        file_path = os.path.join(logs_path, filename)
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))

        QToolTip.showText(self.btn_save_results.mapToGlobal(QPoint(0,0)), f'File saved to {file_path}', self.btn_save_results, QRect(), 1000)
    
    def return_to_parameters(self):
        if self.main_panel.parameters.manager:
            self.main_panel.parameters.manager.stop()
        self.main_panel.general_job.reset()

        self.main_panel.switch_current_job_display(0)
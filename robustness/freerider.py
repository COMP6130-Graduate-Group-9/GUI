from PyQt5.QtCore import QPoint, QProcess, QRect, pyqtSlot, Qt
from PyQt5.QtWidgets import (QGridLayout, QHBoxLayout, QSizePolicy, QStackedLayout, QTableWidget, QTableWidgetItem, QToolTip, QVBoxLayout, QWidget,
    QComboBox, QLabel, QPushButton, QProgressBar, QPlainTextEdit)

import re, os, time
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

        self.algo = "FedAvg"
        self.dataset = "MNIST-iid"
        self.epochs = "5"
        self.type = "FL"
        self.coef = "1"
        self.power = "1"
        self.n_freeriders = "0"

        grid = QGridLayout()
        self.setLayout(grid)

        description = QLabel("Please specify your parameters and submit the job")
        description.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        algo_list = [
            "FedAvg",
            "FedProx"
        ]

        dataset_list = [
            "MNIST-iid",
            "MNIST-shard",
            "CIFAR-10",
            "shakespeare"
        ]

        type_list = [
            "FL",
            "plain",
            "disguised",
            "many"
        ]

        epochs_list = ["5", "20"]
        coef_list = ["1", "3"]
        power_list = ["0.5", "1", "2"]
        n_freeriders_list = ["0", "1", "5", "45"]

        algo_box = QHBoxLayout()
        algo_input = QComboBox()
        for algo in algo_list:
            algo_input.addItem(algo, algo)
        algo_input.currentTextChanged.connect(lambda i: self.text_changed(i, "algo"))
        curr_algo_idx = algo_input.findData(self.algo)
        algo_input.setCurrentIndex(curr_algo_idx)
        algo_label = QLabel("Algorithm:")
        algo_label.setBuddy(algo_input)
        algo_box.addWidget(algo_label)
        algo_box.addWidget(algo_input)

        dataset_box = QHBoxLayout()
        dataset_input = QComboBox()
        for dataset in dataset_list:
            dataset_input.addItem(dataset, dataset)
        dataset_input.currentTextChanged.connect(lambda i: self.text_changed(i, "dataset"))
        curr_dataset_idx = dataset_input.findData(self.dataset)
        dataset_input.setCurrentIndex(curr_dataset_idx)
        dataset_label = QLabel("Dataset:")
        dataset_label.setBuddy(dataset_input)
        dataset_box.addWidget(dataset_label)
        dataset_box.addWidget(dataset_input)

        type_box = QHBoxLayout()
        type_input = QComboBox()
        for type in type_list:
            type_input.addItem(type, type)
        type_input.currentTextChanged.connect(lambda i: self.text_changed(i, "type"))
        curr_type_idx = type_input.findData(self.type)
        type_input.setCurrentIndex(curr_type_idx)
        type_label = QLabel("Experiment type:")
        type_label.setBuddy(type_input)
        type_box.addWidget(type_label)
        type_box.addWidget(type_input)

        epochs_box = QHBoxLayout()
        epochs_input = QComboBox()
        for epochs in epochs_list:
            epochs_input.addItem(epochs, epochs)
        epochs_input.currentTextChanged.connect(lambda i: self.text_changed(i, "epochs"))
        curr_epochs_idx = epochs_input.findData(self.epochs)
        epochs_input.setCurrentIndex(curr_epochs_idx)
        epochs_label = QLabel("Epochs:")
        epochs_label.setBuddy(epochs_input)
        epochs_box.addWidget(epochs_label)
        epochs_box.addWidget(epochs_input)

        coef_box = QHBoxLayout()
        coef_input = QComboBox()
        for coef in coef_list:
            coef_input.addItem(coef, coef)
        coef_input.currentTextChanged.connect(lambda i: self.text_changed(i, "coef"))
        curr_coef_idx = coef_input.findData(self.coef)
        coef_input.setCurrentIndex(curr_coef_idx)
        coef_label = QLabel("Coef:")
        coef_label.setBuddy(coef_input)
        coef_box.addWidget(coef_label)
        coef_box.addWidget(coef_input)

        power_box = QHBoxLayout()
        power_input = QComboBox()
        for power in power_list:
            power_input.addItem(power, power)
        power_input.currentTextChanged.connect(lambda i: self.text_changed(i, "power"))
        curr_power_idx = coef_input.findData(self.power)
        power_input.setCurrentIndex(curr_power_idx)
        power_label = QLabel("Power:")
        power_label.setBuddy(power_input)
        power_box.addWidget(power_label)
        power_box.addWidget(power_input)

        n_freeriders_box = QHBoxLayout()
        n_freeriders_input = QComboBox()
        for n_freeriders in n_freeriders_list:
            n_freeriders_input.addItem(n_freeriders, n_freeriders)
        n_freeriders_input.currentTextChanged.connect(lambda i: self.text_changed(i, "n_freeriders"))
        curr_n_freeriders_idx = n_freeriders_input.findData(self.n_freeriders)
        n_freeriders_input.setCurrentIndex(curr_n_freeriders_idx)
        n_freeriders_label = QLabel("Number of freeriders:")
        n_freeriders_label.setBuddy(n_freeriders_input)
        n_freeriders_box.addWidget(n_freeriders_label)
        n_freeriders_box.addWidget(n_freeriders_input)

        btn_submit = QPushButton("SUBMIT")
        btn_submit.pressed.connect(self.submit_job)

        grid.addWidget(description, 0, 0, 1, 4)
        grid.addLayout(algo_box, 1, 0, 1, 1)
        grid.addLayout(dataset_box, 1, 1, 1, 1)
        grid.addLayout(type_box, 1, 2, 1, 1)
        grid.addLayout(epochs_box, 1, 3, 1, 1)
        grid.addLayout(coef_box, 2, 0, 1, 1)
        grid.addLayout(power_box, 2, 1, 1, 1)
        grid.addLayout(n_freeriders_box, 2, 2, 1, 1)
        grid.addWidget(btn_submit, 3, 0, 1, 4)

    def value_changed(self, i, name):
        self.__dict__[name] = i

    def text_changed(self, text, name):
        setattr(self, name, text)

    def submit_job(self):
        self.main_panel.switch_current_job_display(1)

        if self.dataset == "MNIST-iid":
            folder_name = "MNIST"
            script = "mnist_iid.py"
        elif self.dataset == "MNIST-shard":
            folder_name = "MNIST"
            script = "MIST_non_iid.py"
        elif self.dataset == "CIFAR-10":
            folder_name = "CIFAR-10"
            script = "create_CIFAR.py"
        elif self.dataset == "shakespeare":
            folder_name = "shakespeare"
            script = "shakespeare.py"
        
        exec_dir = f"algorithms/Labs-Federated-Learning/data/{folder_name}"
        return_path = "../../../../"
        self.manager = ProcessManager(exec_dir, return_path)
        self.manager.started.connect(self.main_panel.general_job.generate_dataset_initialize)
        self.manager.finished.connect(self.main_panel.general_job.generate_dataset_complete)
        self.manager.textChanged.connect(self.main_panel.general_job.generate_dataset_update_status)

        self.manager.run_script(script)


class GeneralJob(QWidget):
    def __init__(self, main_panel):
        super().__init__()

        self.main_panel = main_panel
        self.manager = None

        self.status = "running"

        layout = QGridLayout()
        self.setLayout(layout)

        progress_box = QHBoxLayout()
        self.current_action = QLabel()
        self.current_action.setText('Generating dataset distribution.....')
        progress_box.addWidget(self.current_action)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        progress_box.addWidget(self.progress)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)

        self.right_box = QVBoxLayout()
        self.timer = Timer()
        self.right_box.addWidget(self.timer)
        self.btn_extra_action = QPushButton("View result")
        self.btn_extra_action.pressed.connect(self.extra_action)
        self.btn_extra_action.setEnabled(False)
        self.right_box.addWidget(self.btn_extra_action)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.pressed.connect(self.cancel)
        self.right_box.addWidget(self.btn_cancel)

        layout.addLayout(progress_box, 0, 0, 1, 5)
        layout.addWidget(self.text, 1, 0, 1, 1)
        layout.addLayout(self.right_box, 1, 1, 1, 1)

    @pyqtSlot()
    def generate_dataset_initialize(self):
        pass

    @pyqtSlot(str)
    def generate_dataset_update_status(self, status):
        print(status)
        self.text.appendPlainText(status)

    @pyqtSlot(int, QProcess.ExitStatus)
    def generate_dataset_complete(self, exitCode, exitStatus):
        if exitStatus == 0:
            self.run_experiment()

    @pyqtSlot()
    def initialize(self):
        self.current_action.setText("Running experiment......")
        self.progress.setValue(0)
        self.timer.start()
    
    @pyqtSlot(str)
    def update_status(self, status):
        print(status)
        self.track_progress(status)
        self.track_global_result(status)
        self.text.appendPlainText(status)

    @pyqtSlot(int, QProcess.ExitStatus)
    def complete(self, exitCode, exitStatus):
        if exitStatus == 0:
            self.timer.stop()
            self.progress.setValue(100)
            self.btn_cancel.setParent(None)
            self.btn_extra_action.setEnabled(True)
            self.status = "complete"

    def run_experiment(self):
        exec_dir = f"algorithms/Labs-Federated-Learning"
        return_path = "../../"
        self.manager = ProcessManager(exec_dir, return_path)
        self.manager.started.connect(self.initialize)
        self.manager.finished.connect(self.complete)
        self.manager.textChanged.connect(self.update_status)

        algo = self.main_panel.parameters.algo
        dataset = self.main_panel.parameters.dataset
        type = self.main_panel.parameters.type
        epochs = self.main_panel.parameters.epochs
        coef = self.main_panel.parameters.coef
        power = self.main_panel.parameters.power
        n_freeriders = self.main_panel.parameters.n_freeriders
        script = f"free-riding.py --algo {algo} --dataset {dataset} --type {type} --epochs {epochs} --coef {coef} --power {power} --n_freeriders {n_freeriders} --simple_experiment False"
        self.manager.run_script(script)

    def extra_action(self):
        self.main_panel.results.get_latest_elapsed_time()
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

        self.status = "running"
        
        self.current_action.setText('Generating dataset.....')
        self.progress.setValue(0)
        self.btn_extra_action.setEnabled(False)
        self.right_box.addWidget(self.btn_cancel)

    def track_progress(self, status):
        re_iter = re.findall(r'n_iter: (\d+)', status)
        re_progress = re.findall(r'====> i: (\d+) Loss: ([0-9]*[.]?[0-9]+e?[-+]?[\d]+) Server Test Accuracy: ([0-9]*[.]?[0-9]+)', status)
        if len(re_iter) > 0:
            self.total_iter = int(re_iter[0])
        if len(re_progress) > 0:
            curr_iter, loss, accuracy = int(re_progress[0][0]), float(re_progress[0][1]), float(re_progress[0][2])
            self.main_panel.results.record_loss(loss)
            self.main_panel.results.record_accuracy(accuracy)
            percentage = curr_iter / self.total_iter * 100
            self.progress.setValue(percentage)

    def track_global_result(self, status):
        re_accuracy = re.findall(r'Test set: Accuracy: \d+/\d+ \((\d+)%\)', status)
        re_loss = re.findall(r'Test set: Loss: ([0-9]*[.]?[0-9]+)', status)
        if len(re_accuracy) > 0:
            accuracy = int(re_accuracy[0]) / 100
            self.main_panel.results.record_accuracy(accuracy)
        if len(re_loss) > 0:
            loss = float(re_loss[0])
            self.main_panel.results.record_loss(loss)


class Results(QWidget):
    def __init__(self, main_panel):
        super().__init__()

        self.main_panel = main_panel

        self.accuracy = 0
        self.loss = 0
        self.elapsed_time = ""

        layout = QGridLayout()
        self.setLayout(layout)

        numerical_result_box = QVBoxLayout()
        self.accuracy_display = QLabel()
        numerical_result_box.addWidget(self.accuracy_display)
        self.loss_display = QLabel()
        numerical_result_box.addWidget(self.loss_display)
        self.elapsed_time_display = QLabel()
        numerical_result_box.addWidget(self.elapsed_time_display)

        table = QTableWidget(self)
        table.setColumnCount(2)
        table.setRowCount(8)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        table.setSizePolicy(sizePolicy)

        lr = 10 ** -3
        if self.main_panel.parameters.dataset == "shakespeare":
            lr = 0.5
        self.parameters = {
            'Algorithm': self.main_panel.parameters.algo,
            'Dataset': self.main_panel.parameters.dataset,
            'Epochs': self.main_panel.parameters.epochs,
            'Experiment type': self.main_panel.parameters.type,
            'Coef': self.main_panel.parameters.coef,
            'Power': self.main_panel.parameters.power,
            'Number of free riders': self.main_panel.parameters.n_freeriders,
            'Learning rate': lr,
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

    def record_accuracy(self, accuracy):
        self.accuracy = accuracy
        self.accuracy_display.setText(f"Accuracy: {float(self.accuracy)}")

    def record_loss(self, loss):
        self.loss = loss
        self.loss_display.setText(f"Loss: {float(self.loss)}")

    def get_latest_elapsed_time(self):
        self.elapsed_time = self.main_panel.general_job.timer.elapsed_time
        self.elapsed_time_display.setText(f"Elapsed time: {self.elapsed_time}")

    def save_results(self):
        logs_path = os.path.join(os.getcwd(), 'logs')
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)

        lines = [
            'Result',
            f'Accuracy: {self.accuracy}',
            f'Loss: {self.loss}',
            f'Elapsed time: {self.elapsed_time}',
            '',
            'Parameters'
        ]
        lines += [f'{p[0]}: {p[1]}' for p in self.parameters.items()]
        t = time.localtime()
        current_time = time.strftime("%m-%d-%Y_%H-%M-%S", t)
        filename = f'robustness-freerider-{current_time}.txt'
        file_path = os.path.join(logs_path, filename)
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))

        QToolTip.showText(self.btn_save_results.mapToGlobal(QPoint(0,0)), f'File saved to {file_path}', self.btn_save_results, QRect(), 1000)
    
    def return_to_parameters(self):
        if self.main_panel.parameters.manager:
            self.main_panel.parameters.manager.stop()
        self.main_panel.general_job.reset()

        self.main_panel.switch_current_job_display(0)
from PyQt5.QtCore import QPoint, QProcess, QRect, pyqtSlot, Qt
from PyQt5.QtWidgets import (QGridLayout, QHBoxLayout, QSizePolicy, QStackedLayout, QTableWidget, QTableWidgetItem, QToolTip, QVBoxLayout, QWidget,
    QComboBox, QLabel, QPushButton, QProgressBar, QPlainTextEdit,
    QScrollArea)

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

        self.experiment = "Label Flipping Attack Feasibility"
        self.num_of_exp = 3
        self.num_of_clients = 50

        grid = QGridLayout()
        self.setLayout(grid)

        description = QLabel("Please specify your parameters and submit the job")
        description.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        experiment_list = [
            "Label Flipping Attack Feasibility",
            "Attack Timing in Label Flipping Attacks",
            "Malicious Participant Availability"
        ]

        experiment_box = QHBoxLayout()
        experiment_input = QComboBox()
        for experiment in experiment_list:
            experiment_input.addItem(experiment, experiment)
        experiment_input.currentTextChanged.connect(self.experiment_text_changed)
        curr_experiment_idx = experiment_input.findData(self.experiment)
        experiment_input.setCurrentIndex(curr_experiment_idx)
        experiment_label = QLabel("experiment:")
        experiment_label.setBuddy(experiment_input)
        experiment_box.addWidget(experiment_label)
        experiment_box.addWidget(experiment_input)

        btn_submit = QPushButton("SUBMIT")
        btn_submit.pressed.connect(self.submit_job)

        grid.addWidget(description, 0, 0, 1, 1)
        grid.addLayout(experiment_box, 1, 0, 1, 1)
        grid.addWidget(btn_submit, 2, 0, 1, 1)

    def experiment_text_changed(self, experiment):
        self.experiment = experiment

    def submit_job(self):
        self.main_panel.switch_current_job_display(1)

        exec_dir = f"algorithms/DataPoisoning_FL"
        return_path = "../../"
        self.manager = ProcessManager(exec_dir, return_path)
        self.manager.started.connect(self.main_panel.general_job.generate_data_distribution_initialize)
        self.manager.finished.connect(self.main_panel.general_job.generate_data_distribution_complete)
        self.manager.textChanged.connect(self.main_panel.general_job.generate_data_distribution_update_status)

        script = "generate_data_distribution.py"
        self.manager.run_script(script)


class GeneralJob(QWidget):
    def __init__(self, main_panel):
        super().__init__()

        self.main_panel = main_panel

        self.status = "running"
        self.curr_num_of_exp = 1
        self.curr_client = 0
        self.curr_epoch = 0
        self.percentage = 0

        self.layout = QStackedLayout()
        self.setLayout(self.layout)

        main_job = QWidget()
        main_job_layout = QGridLayout()
        main_job.setLayout(main_job_layout)

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
        self.btn_extra_action = QPushButton("View job")
        self.btn_extra_action.pressed.connect(self.extra_action)
        self.right_box.addWidget(self.btn_extra_action)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.pressed.connect(self.cancel)
        self.right_box.addWidget(self.btn_cancel)

        main_job_layout.addLayout(progress_box, 0, 0, 1, 5)
        main_job_layout.addWidget(self.text, 1, 0, 1, 1)
        main_job_layout.addLayout(self.right_box, 1, 1, 1, 1)

        self.individual_job = Clients(main_panel, self)
        
        self.layout.addWidget(main_job)
        self.layout.addWidget(self.individual_job)
        self.layout.setCurrentIndex(0)

    @pyqtSlot()
    def generate_data_distribution_initialize(self):
        pass
    
    @pyqtSlot(str)
    def generate_data_distribution_update_status(self, status):
        print(status)
        self.text.appendPlainText(status)

    @pyqtSlot(int, QProcess.ExitStatus)
    def generate_data_distribution_complete(self, exitCode, exitStatus):
        if exitStatus == 0:
            self.generate_default_models()

    @pyqtSlot()
    def generate_default_models_initialize(self):
        self.current_action.setText("Generating default models......")
    
    @pyqtSlot(str)
    def generate_default_models_update_status(self, status):
        print(status)
        self.text.appendPlainText(status)

    @pyqtSlot(int, QProcess.ExitStatus)
    def generate_default_models_complete(self, exitCode, exitStatus):
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
        self.track_client(status)
        self.track_global_result(status)
        self.text.appendPlainText(status)

    @pyqtSlot(int, QProcess.ExitStatus)
    def complete(self, exitCode, exitStatus):
        if exitStatus == 0:
            self.timer.stop()
            self.progress.setValue(100)
            self.btn_cancel.setParent(None)
            self.btn_extra_action.setText("View results")
            self.status = "complete"
            self.individual_job.complete_all_clients()

    def generate_default_models(self):
        exec_dir = f"algorithms/DataPoisoning_FL"
        return_path = "../../"
        self.manager = ProcessManager(exec_dir, return_path)
        self.manager.started.connect(self.generate_default_models_initialize)
        self.manager.finished.connect(self.generate_default_models_complete)
        self.manager.textChanged.connect(self.generate_default_models_update_status)

        script = "generate_default_models.py"
        self.manager.run_script(script)
    
    def run_experiment(self):
        exec_dir = f"algorithms/DataPoisoning_FL"
        return_path = "../../"
        self.manager = ProcessManager(exec_dir, return_path)
        self.manager.started.connect(self.initialize)
        self.manager.finished.connect(self.complete)
        self.manager.textChanged.connect(self.update_status)

        experiment = self.main_panel.parameters.experiment
        if experiment == "Label Flipping Attack Feasibility":
            script = "label_flipping_attack.py"
        elif experiment == "Attack Timing in Label Flipping Attacks":
            script = "attack_timing.py"
            self.num_of_exp = 1
        elif experiment == "Malicious Participant Availability":
            script = "malicious_participant_availability.py"

        self.manager.run_script(script)

    def extra_action(self):
        if self.status == "running":
            self.layout.setCurrentIndex(1)
        else:
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
        self.curr_num_of_exp = 1
        self.curr_client = 0
        self.curr_epoch = 0
        self.percentage = 0
        
        self.current_action.setText('Generating dataset distribution.....')
        self.progress.setValue(0)
        self.right_box.addWidget(self.btn_cancel)
        self.btn_extra_action.setText("View job")
        self.individual_job.reset_all_clients()

    def track_progress(self, status):
        re_epoch = re.findall(r'Training epoch #(\d+) on client #(\d+)', status)
        if len(re_epoch) > 0:
            epoch, self.curr_client = int(re_epoch[0][0]), int(re_epoch[0][1])
            if epoch == 1 and self.curr_epoch == 10:
                self.curr_num_of_exp += 1
            self.curr_epoch = epoch
            self.percentage = (self.curr_epoch / 10 + self.curr_num_of_exp - 1) * 100 / self.main_panel.parameters.num_of_exp
            self.progress.setValue(self.percentage)

    def track_client(self, status):
        re_losses = re.findall(r'\[(\d+),\s+100\] loss: ([0-9]*[.]?[0-9]+)', status)
        if len(re_losses) > 0:
            loss = float(re_losses[0][1])
            client_name = f"Client {self.curr_client}"
            if client_name in self.individual_job.clients:
                self.individual_job.clients[client_name].update(self.percentage, loss)

    def track_global_result(self, status):
        re_accuracy = re.findall(r'Test set: Accuracy: \d+/\d+ \((\d+)%\)', status)
        re_loss = re.findall(r'Test set: Loss: ([0-9]*[.]?[0-9]+)', status)
        if len(re_accuracy) > 0:
            accuracy = int(re_accuracy[0]) / 100
            self.main_panel.results.record_accuracy(accuracy)
        if len(re_loss) > 0:
            loss = float(re_loss[0])
            self.main_panel.results.record_loss(loss)

class Clients(QWidget):
    def __init__(self, main_panel, general_job):
        super().__init__()

        self.main_panel = main_panel
        self.general_job = general_job

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        scroll = QScrollArea()
        self.layout.addWidget(scroll)

        btn_back = QPushButton("Back")
        btn_back.clicked.connect(self.back_to_main_job)
        self.layout.addWidget(btn_back)
    
        wrapper = QWidget()
        grid = QGridLayout()
        wrapper.setLayout(grid)

        self.clients = {}

        for i in range(self.main_panel.parameters.num_of_clients):
            client_name = f"Client {i+1}"
            object = self.IndividualClient(client_name)
            self.clients[client_name] = object
            grid.addWidget(object, i//5, i%5)

        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(wrapper)

    def back_to_main_job(self):
        self.general_job.layout.setCurrentIndex(0)

    def reset_all_clients(self):
        for c in self.clients.values():
            c.reset()

    def complete_all_clients(self):
        for c in self.clients.values():
            c.complete()

    class IndividualClient(QWidget):
        def __init__(self, name):
            super().__init__()

            self.layout = QVBoxLayout()
            self.setLayout(self.layout)

            client_name = QLabel(name)
            self.progress = QProgressBar()
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.loss = QLabel("Local Loss: 0")
            
            self.layout.addWidget(client_name)
            self.layout.addWidget(self.progress)
            self.layout.addWidget(self.loss)

        def update(self, progress, loss):
            self.progress.setValue(progress)
            self.loss.setText(f"Local Loss: {float(loss):.3f}")

        def reset(self):
            self.progress.setValue(0)
            self.loss.setText("Local Loss: 0")

        def complete(self):
            self.progress.setValue(100)

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
        table.setRowCount(7)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        table.setSizePolicy(sizePolicy)

        self.parameters = {
            'Dataset': "Fashion-MNIST",
            'Number of clients': self.main_panel.parameters.num_of_clients,
            'Learning rate': '0.01',
            'Epochs': '10',
            'Batch size': '10',
            'Scheduler step size': '50',
            'Scheduler gamma': '0.5',
            'Min. learning rate': '1e-10',
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
        self.accuracy_display.setText(f"Accuracy: {float(self.accuracy):.4f}")

    def record_loss(self, loss):
        self.loss = loss
        self.loss_display.setText(f"Loss: {float(self.loss):.2f}")

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
        filename = f'robustness-data-poisoning-exp_{self.main_panel.parameters.experiment}-{current_time}.txt'
        file_path = os.path.join(logs_path, filename)
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))

        QToolTip.showText(self.btn_save_results.mapToGlobal(QPoint(0,0)), f'File saved to {file_path}', self.btn_save_results, QRect(), 1000)
    
    def return_to_parameters(self):
        if self.main_panel.parameters.manager:
            self.main_panel.parameters.manager.stop()
        self.main_panel.general_job.reset()

        self.main_panel.switch_current_job_display(0)
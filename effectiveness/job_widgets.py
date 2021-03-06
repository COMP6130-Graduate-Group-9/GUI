from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import QPoint, QTimer, pyqtSlot, QProcess, Qt, QRect
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QScrollArea, QStackedLayout, QTableWidgetItem, QWidget, QVBoxLayout, QProgressBar, QPlainTextEdit, QGridLayout,
    QHBoxLayout, QSpinBox, QComboBox, QLabel, QDoubleSpinBox,
    QPushButton, QFrame, QSizePolicy, QTableWidget, QToolTip)

import re, os, time
from util.process_manager import ProcessManager
from util.timer import Timer

class QHSeperationLine(QFrame):
  '''
  a horizontal seperation line\n
  '''
  def __init__(self):
    super().__init__()
    self.setMinimumWidth(1)
    self.setFixedHeight(20)
    self.setFrameShape(QFrame.HLine)
    self.setFrameShadow(QFrame.Sunken)
    self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    return

class Parameters(QWidget):
    def __init__(self, main_panel):
        super().__init__()

        self.main_panel = main_panel

        self.dataset = "Mnist"
        self.sampling_ratio = 0.5
        self.alpha = 0.1
        self.num_of_clients = 3
        self.learning_rate = 0.01
        self.global_iterations = 100
        self.total_epochs = 20

        layout = QVBoxLayout()
        self.setLayout(layout)
        
        description = QLabel("Please specify your parameters and submit the job")
        description.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
        layout.addWidget(description)
        grid = QGridLayout()
        layout.addLayout(grid)

        btn_submit = QPushButton("SUBMIT")
        btn_submit.pressed.connect(self.submit_job)
        layout.addWidget(btn_submit)

        dataset_box = QHBoxLayout()
        dataset_input = QComboBox()
        dataset_input.addItems(["MNIST", "EMINST", "CELEBA"])
        dataset_input.currentTextChanged.connect(self.text_changed)
        dataset_label = QLabel("Dataset:")
        dataset_label.setBuddy(dataset_input)
        dataset_box.addWidget(dataset_label)
        dataset_box.addWidget(dataset_input)

        alpha_box = QHBoxLayout()
        alpha_input = QDoubleSpinBox()
        alpha_input.setValue(self.alpha)
        alpha_input.valueChanged.connect(lambda i: self.value_changed(i, "alpha"))
        alpha_label = QLabel("Alpha:")
        alpha_label.setBuddy(alpha_input)
        alpha_box.addWidget(alpha_label)
        alpha_box.addWidget(alpha_input)

        sampling_ratio_box = QHBoxLayout()
        sampling_ratio_input = QDoubleSpinBox()
        sampling_ratio_input.setValue(self.sampling_ratio)
        sampling_ratio_input.valueChanged.connect(lambda i: self.value_changed(i, "sampling_ratio"))
        sampling_ratio_label = QLabel("Sampling Ratio:")
        sampling_ratio_label.setBuddy(sampling_ratio_input)
        sampling_ratio_box.addWidget(sampling_ratio_label)
        sampling_ratio_box.addWidget(sampling_ratio_input)
        
        dataset_param_box = QHBoxLayout()
        dataset_param_box.addLayout(alpha_box)
        dataset_param_box.addLayout(sampling_ratio_box)

        num_of_clients_box = QHBoxLayout()
        num_of_clients_input = QSpinBox()
        num_of_clients_input.setValue(self.num_of_clients)
        num_of_clients_input.valueChanged.connect(lambda i: self.value_changed(i, "num_of_clients"))
        num_of_clients_label = QLabel("Number of Clients:")
        num_of_clients_label.setBuddy(num_of_clients_input)
        num_of_clients_box.addWidget(num_of_clients_label)
        num_of_clients_box.addWidget(num_of_clients_input)
        
        learning_rate_box = QHBoxLayout()
        learning_rate_input = QDoubleSpinBox()
        learning_rate_input.setValue(self.learning_rate)
        learning_rate_input.valueChanged.connect(lambda i: self.value_changed(i, "learning_rate"))
        learning_rate_label = QLabel("Learning Rate:")
        learning_rate_label.setBuddy(learning_rate_input)
        learning_rate_box.addWidget(learning_rate_label)
        learning_rate_box.addWidget(learning_rate_input)

        global_iterations_box = QHBoxLayout()
        global_iterations_input = QSpinBox()
        global_iterations_input.setMaximum(1000)
        global_iterations_input.setValue(self.global_iterations)
        global_iterations_input.valueChanged.connect(lambda i: self.value_changed(i, "global_iterations"))
        global_iterations_label = QLabel("Global Iterations:")
        global_iterations_label.setBuddy(global_iterations_input)
        global_iterations_box.addWidget(global_iterations_label)
        global_iterations_box.addWidget(global_iterations_input)

        total_epochs_box = QHBoxLayout()
        total_epochs_input = QSpinBox()
        total_epochs_input.setValue(self.total_epochs)
        total_epochs_input.valueChanged.connect(lambda i: self.value_changed(i, "total_epochs"))
        total_epochs_label = QLabel("Total epochs:")
        total_epochs_label.setBuddy(total_epochs_input)
        total_epochs_box.addWidget(total_epochs_label)
        total_epochs_box.addWidget(total_epochs_input)

        seperator_horizontal = QHSeperationLine()

        grid.addLayout(dataset_box, 0, 0, 2, 1)
        grid.addLayout(dataset_param_box, 0, 1, 2, 1)
        grid.addWidget(seperator_horizontal, 1, 0, 2, 4)
        grid.addLayout(num_of_clients_box, 2, 0, 2, 1)
        grid.addLayout(learning_rate_box, 2, 1, 2, 1)
        grid.addLayout(global_iterations_box, 2, 2, 2, 1)
        grid.addLayout(total_epochs_box, 2, 3, 2, 1)

    def value_changed(self, i, name):
        self.__dict__[name] = i

    def text_changed(self, s):
        if s == "MNIST":
            self.dataset = "Mnist"
        elif s == "EMNIST":
            self.dataset = "EMnist"
        else:
            self.dataset = "Celeb"

    def submit_job(self):
        self.main_panel.switch_current_job_display(1)

        exec_dir = f"algorithms/FedGen/data/{self.dataset}"
        return_path = "../../../../"
        self.manager = ProcessManager(exec_dir, return_path)
        self.manager.started.connect(self.main_panel.general_job_container.initialize)
        self.manager.finished.connect(self.main_panel.general_job_container.dataset_complete)
        self.manager.textChanged.connect(self.main_panel.general_job_container.update_status)

        if self.dataset == "Celeb":
            self.main_panel.general_job_container.skip_dataset()
        else:
            script = f"generate_niid_dirichlet.py --n_class 10 --sampling_ratio {self.sampling_ratio} --alpha {self.alpha} --n_user 20"
            self.manager.run_script(script)

class GeneralJob(QWidget):
    def __init__(self, main_panel) -> None:
        super().__init__()

        self.main_panel = main_panel
        self.iteration = 0
        self.round = 0
        self.manager = None
        self.status = "running"

        self.layout = QStackedLayout()
        self.setLayout(self.layout)

        main_job = QWidget()
        main_job_layout = QGridLayout()
        main_job.setLayout(main_job_layout)
        
        progress_box = QHBoxLayout()
        self.current_action = QLabel()
        self.current_action.setText('Generating dataset.....')
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

        main_job_layout.addLayout(progress_box, 0, 0, 1, 2)
        main_job_layout.addWidget(self.text, 1, 0, 1, 1)
        main_job_layout.addLayout(self.right_box, 1, 1, 1, 1)

        self.individual_job = Clients(main_panel)
        
        self.layout.addWidget(main_job)
        self.layout.addWidget(self.individual_job)
        self.layout.setCurrentIndex(0)
    
    @pyqtSlot()
    def initialize(self):
        self.progress.setValue(0)
        self.main_panel.results_container.create_table()
    
    @pyqtSlot(str)
    def update_status(self, status):
        print(status)
        self.text.appendPlainText(status)

    @pyqtSlot(str)
    def experiment_update_status(self, status):
        percentage = self.track_progress(status)
        self.track_client(status, percentage)
        self.track_global_result(status)
        self.text.appendPlainText(status)

    @pyqtSlot(int, QProcess.ExitStatus)
    def dataset_complete(self, exitCode, exitStatus):
        if exitStatus == 0:
            self.progress.setValue(100)
            self.run_experiment()

    @pyqtSlot(int, QProcess.ExitStatus)
    def complete(self, exitCode, exitStatus):
        if exitStatus == 0:
            self.timer.stop()
            self.progress.setValue(100)
            self.current_action.setText('Experiment complete')
            self.btn_cancel.setParent(None)
            self.btn_extra_action.setText("View results")
            self.status = "complete"
            self.individual_job.complete_all_clients()

    def skip_dataset(self):
        self.progress.setValue(100)
        self.run_experiment()

    def run_experiment(self):
        self.current_action.setText('Running experiment.....')
        
        exec_dir = "algorithms/FedGen"
        return_path = "../../"
        self.manager = ProcessManager(exec_dir, return_path)
        self.manager.started.connect(self.main_panel.general_job_container.initialize)
        self.manager.finished.connect(self.main_panel.general_job_container.complete)
        self.manager.textChanged.connect(self.main_panel.general_job_container.experiment_update_status)

        alpha = self.main_panel.parameters_container.alpha
        ratio = self.main_panel.parameters_container.sampling_ratio
        learning_rate = self.main_panel.parameters_container.learning_rate
        clients = self.main_panel.parameters_container.num_of_clients
        global_iterations = self.main_panel.parameters_container.global_iterations
        total_epochs = self.main_panel.parameters_container.total_epochs
        script = f"main.py --dataset Mnist-alpha{alpha}-ratio{ratio} --algorithm FedGen --batch_size 32 --num_glob_iters {global_iterations} --local_epochs {total_epochs} --num_users {clients} --lamda 1 --learning_rate {learning_rate} --model cnn --personal_learning_rate 0.01 --times 3"
        self.manager.run_script(script)
        self.timer.start()

    def track_progress(self, status):
        updated = False
        re_iteration = re.findall(r'Start training iteration \d+', status)
        if len(re_iteration) > 0:
            iteration_matched_text = re_iteration[-1]
            new_iteration = int(re.findall(r'\d+', iteration_matched_text)[0])
            if new_iteration != self.iteration and round == 99:
                self.round = 0
            self.iteration = new_iteration
            updated = True
        re_round = re.findall(r'Round number:  \d+', status)
        if re_round:
            round_matched_text = re_round[-1]
            self.round = int(re.findall(r'\d+', round_matched_text)[0])
            updated = True
        
        if updated:
            total_rounds = self.main_panel.parameters_container.global_iterations
            total_iterations = 3
            percentage = self.iteration * (100 / total_iterations) + self.round / total_rounds * (100 / total_iterations)
            self.progress.setValue(percentage)
            return percentage
        return None

    def track_client(self, status, percentage):
        re_client = re.findall(r'Client (\d+):', status)
        re_accuracy = re.findall(r'accuracy ([0-9]*[.]?[0-9]+)', status)
        re_losses = re.findall(r'loss ([0-9]*[.]?[0-9]+)', status)
        for client, accuracy, loss in zip(re_client, re_accuracy, re_losses):
            client_name = f"Client {client}"
            if client_name in self.individual_job.clients:
                self.individual_job.clients[client_name].update(percentage, accuracy, loss)

    def track_global_result(self, status):
        re_global_result = re.findall(r'Average Global Accurancy = ([0-9]*[.]?[0-9]+), Loss = ([0-9]*[.]?[0-9]+).', status)
        if len(re_global_result) > 0:
            latest = re_global_result[-1]
            self.main_panel.results_container.record_global_result(latest[0], latest[1])

    def extra_action(self):
        if self.status == "running":
            self.layout.setCurrentIndex(1)
        else:
            self.main_panel.results_container.get_latest_elapsed_time()
            self.main_panel.switch_current_job_display(2)

    def cancel(self):
        if self.main_panel.parameters_container.manager:
            self.main_panel.parameters_container.manager.stop()
        self.reset()

        self.main_panel.switch_current_job_display(0)

    def reset(self):
        if self.manager:
            self.manager.stop()
        self.timer.stop()
        self.timer.reset()

        self.status = "running"
        self.text.clear()
        self.current_action.setText('Generating dataset.....')
        self.progress.setValue(0)
        self.right_box.addWidget(self.btn_cancel)
        self.btn_extra_action.setText("View job")
        self.individual_job.reset_all_clients()

class Clients(QWidget):
    def __init__(self, main_panel):
        super().__init__()

        self.main_panel = main_panel

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

        for i in range(self.main_panel.parameters_container.num_of_clients):
            client_name = f"Client {i+1}"
            object = self.IndividualClient(client_name)
            self.clients[client_name] = object
            grid.addWidget(object, i//5, i%5)

        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(wrapper)

    def back_to_main_job(self):
        self.main_panel.general_job_container.layout.setCurrentIndex(0)

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
            self.accuracy = QLabel("Local accuracy: 0")
            self.loss = QLabel("Local Loss: 0")
            
            self.layout.addWidget(client_name)
            self.layout.addWidget(self.progress)
            self.layout.addWidget(self.accuracy)
            self.layout.addWidget(self.loss)

        def update(self, progress, accuracy, loss):
            self.progress.setValue(progress)
            self.accuracy.setText(f"Local accuracy: {float(accuracy):.4f}")
            self.loss.setText(f"Local Loss: {float(loss):.2f}")

        def reset(self):
            self.progress.setValue(0)
            self.accuracy.setText("Local accuracy: 0")
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

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        numerical_result_box = QVBoxLayout()
        self.accuracy_display = QLabel()
        numerical_result_box.addWidget(self.accuracy_display)
        self.loss_display = QLabel()
        numerical_result_box.addWidget(self.loss_display)
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
        table.setRowCount(7)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        table.setSizePolicy(sizePolicy)

        self.parameters = {
            'Dataset': self.main_panel.parameters_container.dataset,
            'Sampling ratio': self.main_panel.parameters_container.sampling_ratio,
            'Alpha': self.main_panel.parameters_container.alpha,
            'Number of clients': self.main_panel.parameters_container.num_of_clients,
            'Learning rate': self.main_panel.parameters_container.learning_rate,
            'Global iterations': self.main_panel.parameters_container.global_iterations,
            'Total epochs': self.main_panel.parameters_container.total_epochs
        }

        for idx, val in enumerate(self.parameters.items()):
            table.setItem(idx, 0, QTableWidgetItem(val[0]))
            table.setItem(idx, 1, QTableWidgetItem(str(val[1])))

        table.resizeColumnsToContents()
        table.resizeRowsToContents()

        self.layout.addWidget(table, 0, 1, 1, 1)

    
    def record_global_result(self, accuracy, loss):
        self.accuracy = accuracy
        self.accuracy_display.setText(f"Accuracy: {float(self.accuracy):.4f}")
        self.loss = loss
        self.loss_display.setText(f"Loss: {float(self.loss):.2f}")

    def get_latest_elapsed_time(self):
        self.elapsed_time = self.main_panel.general_job_container.timer.elapsed_time
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
        filename = f'effectiveness-{current_time}.txt'
        file_path = os.path.join(logs_path, filename)
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))

        QToolTip.showText(self.btn_save_results.mapToGlobal(QPoint(0,0)), f'File saved to {file_path}', self.btn_save_results, QRect(), 1000)
    
    def return_to_parameters(self):
        if self.main_panel.parameters_container.manager:
            self.main_panel.parameters_container.manager.stop()
        self.main_panel.general_job_container.reset()

        self.main_panel.switch_current_job_display(0)
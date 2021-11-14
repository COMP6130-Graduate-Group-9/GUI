from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QProgressBar, QPlainTextEdit, QGridLayout,
    QHBoxLayout, QSpinBox, QComboBox, QLabel, QDoubleSpinBox,
    QPushButton, QFrame, QSizePolicy)

import re
from process_manager import ProcessManager

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

        grid.addLayout(dataset_box, 0, 0, 1, 1)
        grid.addLayout(dataset_param_box, 1, 0, 1, 1)
        grid.addWidget(seperator_horizontal, 2, 0, 1, 1)
        grid.addLayout(num_of_clients_box, 3, 0)
        grid.addLayout(learning_rate_box, 3, 1)
        grid.addLayout(global_iterations_box, 3, 2)
        grid.addLayout(total_epochs_box, 3, 3)

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
        self.main_panel.switch_to_general_job()

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

        general_job_layout = QVBoxLayout()
        self.setLayout(general_job_layout)
        
        progress_box = QHBoxLayout()
        self.current_action = QLabel()
        self.current_action.setText('Generating dataset.....')
        progress_box.addWidget(self.current_action)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        progress_box.addWidget(self.progress)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)

        general_job_layout.addLayout(progress_box)
        general_job_layout.addWidget(self.text)
    
    @pyqtSlot()
    def initialize(self):
        self.progress.setValue(0)
    
    @pyqtSlot(str)
    def update_status(self, status):
        self.text.appendPlainText(status)

    @pyqtSlot(str)
    def experiment_update_status(self, status):
        self.track_progress(status)
        self.text.appendPlainText(status)

    @pyqtSlot()
    def dataset_complete(self):
        self.progress.setValue(100)
        self.run_experiment()

    @pyqtSlot()
    def complete(self):
        self.progress.setValue(100)
        self.current_action.setText('Experiment complete')

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

    def track_progress(self, status):
        updated = False
        re_iteration = re.findall(r'Start training iteration \d+', status)
        if len(re_iteration) > 0:
            iteration_matched_text = re_iteration[-1]
            self.iteration = int(re.findall(r'\d+', iteration_matched_text)[0])
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

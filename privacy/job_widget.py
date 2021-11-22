from PyQt5.QtCore import QTimer, pyqtSlot, QProcess
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QVBoxLayout, QWidget, QGridLayout, QSpinBox, QLabel, QHBoxLayout,
    QComboBox, QPushButton, QProgressBar, QPlainTextEdit)

from process_manager import ProcessManager

class Parameters(QWidget):
    def __init__(self, main_panel):
        super().__init__()

        self.main_panel = main_panel

        self.model = "ResNet20-4"
        self.dataset = "CIFAR10"
        self.cost_fn = "sim"
        self.indices = "def"
        self.restarts = 1
        self.target_id = -1

        model_list = [
            "ConvNet", "ConvNet8", "ConvNet16", "ConvNet32",
            "BeyondInferringMNIST", "BeyondInferringCifar",
            "MLP", "TwoLP", "ResNet20", "ResNet20-nostride", "ResNet20-10",
            "ResNet20-4", "ResNet20-4-unpooled", "ResNet28-10",
            "ResNet32", "ResNet32-10", "ResNet44", "ResNet56", "ResNe110",
            "ResNet18", "ResNet34", "ResNet50", "ResNet50-2",
            "ResNet101", "ResNet152", "MobileNet", "MNASNet",
            "DenseNet121", "DenseNet40", "DenseNet40-4", 
            "SRNet3", "SRNet1", "iRevNet", "LetNetZhu"]
        dataset_list = ["CIFAR10", "CIFAR100", "MNIST", "MNIST_GRAY", "ImageNet", "BSDS-SR", "BSDS-DN","BSDS-RGB"]

        grid = QGridLayout()  
        self.setLayout(grid)

        description = QLabel("Please specify your parameters and submit the job")

        model_box = QHBoxLayout()
        model_input = QComboBox()
        for model in model_list:
            model_input.addItem(model, model)
        model_input.currentTextChanged.connect(self.model_text_changed)
        curr_model_idx = model_input.findData(self.model)
        model_input.setCurrentIndex(curr_model_idx)
        model_label = QLabel("Model:")
        model_label.setBuddy(model_input)
        model_box.addWidget(model_label)
        model_box.addWidget(model_input)

        dataset_box = QHBoxLayout()
        dataset_input = QComboBox()
        for dataset in dataset_list:
            dataset_input.addItem(dataset, dataset)
        dataset_input.currentTextChanged.connect(self.dataset_text_changed)
        curr_dataset_idx = dataset_input.findData(self.dataset)
        dataset_input.setCurrentIndex(curr_dataset_idx)
        dataset_label = QLabel("Dataset:")
        dataset_label.setBuddy(dataset_input)
        dataset_box.addWidget(dataset_label)
        dataset_box.addWidget(dataset_input)

        cost_fn_box = QHBoxLayout()
        cost_fn_input = QComboBox()
        cost_fn_input.addItem("sim")
        #cost_fn_input.currentTextChanged.connect(self.text_changed)
        cost_fn_label = QLabel("Cost_fn:")
        cost_fn_label.setBuddy(cost_fn_input)
        cost_fn_box.addWidget(cost_fn_label)
        cost_fn_box.addWidget(cost_fn_input)

        indices_box = QHBoxLayout()
        indices_input = QComboBox()
        indices_input.addItem("def")
        #indices_input.currentTextChanged.connet(self.text_changed)
        indices_label = QLabel("Indices:")
        indices_label.setBuddy(indices_input)
        indices_box.addWidget(indices_label)
        indices_box.addWidget(indices_input)

        restarts_box = QHBoxLayout()
        restarts_input = QSpinBox()
        restarts_input.setValue(self.restarts)
        restarts_input.valueChanged.connect(lambda i: self.value_changed(i, "restarts"))
        restarts_label = QLabel("Restarts:")
        restarts_label.setBuddy(restarts_input)
        restarts_box.addWidget(restarts_label)
        restarts_box.addWidget(restarts_input)

        target_id_box = QHBoxLayout()
        target_id_input = QSpinBox()
        target_id_input.setMinimum(-10)
        target_id_input.setValue(self.target_id)
        target_id_input.valueChanged.connect(lambda i: self.value_changed(i, "target_id"))
        target_id_label = QLabel("Target_id:")
        target_id_label.setBuddy(target_id_input)
        target_id_box.addWidget(target_id_label)
        target_id_box.addWidget(target_id_input)

        btn_submit = QPushButton("SUBMIT")
        btn_submit.pressed.connect(self.submit_job)

        grid.addWidget(description, 0, 0, 1, 1)
        grid.addLayout(model_box, 1, 0, 1, 1)
        grid.addLayout(dataset_box, 1, 1, 1, 1)
        grid.addLayout(cost_fn_box, 2, 0, 1, 1)
        grid.addLayout(indices_box, 2, 1, 1, 1)
        grid.addLayout(restarts_box, 3, 0, 1,1 )
        grid.addLayout(target_id_box, 3, 1, 1, 1)
        grid.addWidget(btn_submit, 4, 0, 1, 2)

    def value_changed(self, i, name):
        self.__dict__[name] = i

    def model_text_changed(self, model):
        if model == "ConvNet":
            self.model = "ConvNet"
        elif model == "ConvNet8":
            self.model = "ConvNet8"
        elif model == "ConvNet16":
            self.model = "ConvNet16"
        elif model == "ConvNet32":
            self.model = "ConvNet32"
        elif model == "BeyondInferringMNIST":
            self.model = "BeyondInferringMNIST"
        elif model == "BeyondInferringCifar":
            self.model = "BeyondInferringCifar"
        elif model == "MLP":
            self.model = "MLP"
        elif model == "TwoLP":
            self.model = "TwoLP"
        elif model == "ResNet20":
            self.model = "ResNet20"
        elif model == "ResNet20-nostride":
            self.model = "ResNet20-nostride"
        elif model == "ResNet20-10":
            self.model = "ResNet20-10"
        elif model == "ResNet20-4":
            self.model = "ResNet20-4"
        elif model == "ResNet20-4-unpooled":
            self.model = "ResNet20-4-unpooled"
        elif model == "ResNet28-10":
            self.model = "ResNet28-10"
        elif model == "ResNet32":
            self.model = "ResNet32"
        elif model == "ResNet32-10":
            self.model = "ResNet32-10"
        elif model == "ResNet44":
            self.model = "ResNet44"
        elif model == "ResNet56":
            self.model = "ResNet56"
        elif model == "ResNe110":
            self.model = "ResNe110"
        elif model == "ResNet18":
            self.model = "ResNet18"
        elif model == "ResNet34":
            self.model = "ResNet34"
        elif model == "ResNet50":
            self.model = "ResNet50"
        elif model == "ResNet50-2":
            self.model = "ResNet50-2"
        elif model == "ResNet101":
            self.model = "ResNet101"
        elif model == "ResNet152":
            self.model = "ResNet152"
        elif model == "MobileNet":
            self.model = "MobileNet"
        elif model == "MNASNet":
            self.model = "MNASNet"
        elif model == "DenseNet121":
            self.model = "DenseNet121"
        elif model == "DenseNet40":
            self.model = "DenseNet40"
        elif model == "DenseNet40-4":
            self.model = "DenseNet40-4"
        elif model == "SRNet3":
            self.model = "SRNet3"       
        elif model == "SRNet1":
            self.model = "SRNet1"
        elif model == "iRevNet":
            self.model = "iRevNet"
        elif model == "LetNetZhu":
            self.model = "LetNetZhu"

    def dataset_text_changed(self, dataset):
        if dataset == "CIFAR10":
            self.dataset = "CIFAR10"
        elif dataset == "CIFAR100":
            self.dataset = "CIFAR100"
        elif dataset == "MNIST":
            self.dataset = "MNIST"
        elif dataset == "MNIST_GRAY":
            self.dataset = "MNIST_GRAY"
        elif dataset == "ImageNet":
            self.dataset = "ImageNet"
        elif dataset == "BSDS-SR":
            self.dataset = "BSDS-SR"
        elif dataset == "BSDS-DN":
            self.dataset = "BSDS-DN"
        elif dataset == "BSDS-RGB":
            self.dataset = "BSDS-RGB"

    def submit_job(self):
        self.main_panel.switch_current_job_display(1)

        exec_dir = f"algorithms/invertingGradients"
        return_path = "../../"
        self.manager = ProcessManager(exec_dir, return_path)
        self.manager.started.connect(self.main_panel.general_job_container.initialize)
        self.manager.finished.connect(self.main_panel.general_job_container.complete)
        self.manager.textChanged.connect(self.main_panel.general_job_container.update_status)

        model = self.model
        dataset = self.dataset
        cost_fn = self.cost_fn
        indices = self.indices
        restarts = self.restarts
        target_id = self.target_id
        script = f"reconstruct_image.py --model {model} --dataset {dataset} --trained_model --cost_fn {cost_fn} --indices {indices} --restarts {restarts} --save_image --target_id {target_id}"
        self.manager.run_script(script)

class GeneralJob(QWidget):
    def __init__(self, main_panel):
        super().__init__()

        self.main_panel = main_panel
        self.iteration = 0

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
        layout.addWidget(self.timer, 1, 0, 1, 1)
        layout.addWidget(self.text, 1, 1, 1, 4)

    @pyqtSlot()
    def initialize(self):
        self.progress.setValue(0)
    
    @pyqtSlot(str)
    def update_status(self, status):
        print(status)
        self.text.appendPlainText(status)

    @pyqtSlot(int, QProcess.ExitStatus)
    def complete(self, exitCode, exitStatus):
        if exitStatus == 0:
            self.timer.stop()
            self.progress.setValue(100)
            self.btn_cancel.setParent(None)
            self.btn_view_results.setEnabled(True)

    def view_results(self):
        pass

    def cancel(self):
        if self.manager:
            self.manager.stop()
            self.timer.stop()
            self.timer.reset()

        self.text.clear()
        self.progress.setValue(0)

        self.main_panel.switch_current_job_display(0)

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
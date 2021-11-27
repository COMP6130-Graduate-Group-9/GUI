from PyQt5.QtCore import QTimer, pyqtSlot, QProcess, QRect, QPoint
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (QVBoxLayout, QWidget, QGridLayout, QSpinBox, QLabel, QHBoxLayout,
    QComboBox, QPushButton, QProgressBar, QPlainTextEdit, QToolTip)

import re, os, time
from util.process_manager import ProcessManager
from util.timer import Timer

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
        self.rec_loss = None
        self.output_filename = ''
        self.ground_truth_filename = ''

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
        self.track_image_location(status)
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
        self.main_panel.switch_current_job_display(2)

    def cancel(self):
        if self.main_panel.parameters_container.manager:
            self.main_panel.parameters_container.manager.stop()
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
        re_iteration = re.findall(r'It: (\d+)\.', status)
        re_rec_loss = re.findall(r'Rec. loss: ([0-9]*[.]?[0-9]+)', status)
        if len(re_iteration) > 0:
            percentage = int(re_iteration[0]) / 3_000 * 100
            self.progress.setValue(percentage)
        if len(re_rec_loss) > 0:
            self.rec_loss = re_rec_loss[0]

    def track_image_location(self, status):
        re_output_filename = re.findall(r'Output image saved at: images/(.*\.png) ', status)
        re_ground_truth_filename = re.findall(r'Ground truth image saved at: images/(.*\.png) ', status)
        if len(re_output_filename) > 0:
            self.output_filename = re_output_filename[0]
        if len(re_ground_truth_filename) > 0:
            self.ground_truth_filename = re_ground_truth_filename[0]


class Results(QWidget):
    def __init__(self, main_panel):
        super().__init__()

        self.main_panel = main_panel
        self.rec_loss = None
        self.elapsed_time = None
        self.output_filename = None
        self.ground_truth_filename = None
        self.parameters = {
            'model': main_panel.parameters_container.model,
            'dataset': main_panel.parameters_container.dataset,
            'cost_fn': main_panel.parameters_container.cost_fn,
            'indices': main_panel.parameters_container.indices,
            'restarts': main_panel.parameters_container.restarts,
            'target_id': main_panel.parameters_container.target_id
        }

        layout = QGridLayout()
        self.setLayout(layout)

        ground_truth_title = QLabel("Ground truth")
        self.ground_truth_image = QLabel()

        output_title = QLabel("Output")
        self.output_image = QLabel()

        self.elapsed_time_display = QLabel()
        self.reconstruct_loss_display = QLabel()

        self.btn_save_results = QPushButton("Save results")
        self.btn_save_results.clicked.connect(self.save_results)
        btn_restart = QPushButton("Restart")
        btn_restart.clicked.connect(self.restart)

        layout.addWidget(ground_truth_title, 0, 0, 1, 1)
        layout.addWidget(output_title, 0, 1, 1, 1)
        layout.addWidget(self.ground_truth_image, 1, 0, 1, 1)
        layout.addWidget(self.output_image, 1, 1, 1, 1)
        layout.addWidget(self.elapsed_time_display, 2, 0, 1, 1)
        layout.addWidget(self.reconstruct_loss_display, 2, 1, 1, 1)
        layout.addWidget(self.btn_save_results, 3, 0, 1, 1)
        layout.addWidget(btn_restart, 3, 1, 1, 1)

    def save_results(self):
        logs_path = os.path.join(os.getcwd(), 'logs')
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)

        lines = [
            'Result',
            f'Recontruction Loss: {self.rec_loss}',
            f'Elapsed time: {self.elapsed_time}',
            f'Ground truth image saved to: algorithms/invertingGradients/images/{self.ground_truth_filename}',
            f'Output image saved to: algorithms/invertingGradients/images/{self.output_filename}',
            '',
            'Parameters'
        ]
        lines += [f'{p[0]}: {p[1]}' for p in self.parameters.items()]
        t = time.localtime()
        current_time = time.strftime("%m-%d-%Y_%H-%M-%S", t)
        filename = f'privacy-{current_time}.txt'
        file_path = os.path.join(logs_path, filename)
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))

        QToolTip.showText(self.btn_save_results.mapToGlobal(QPoint(0,0)), f'File saved to {file_path}', self.btn_save_results, QRect(), 1000)

    def restart(self):
        if self.main_panel.parameters_container.manager:
            self.main_panel.parameters_container.manager.stop()
        self.main_panel.general_job_container.reset()

        self.main_panel.switch_current_job_display(0)

    def populate_content(self):
        ground_truth_pixmap = QPixmap(f"algorithms/invertingGradients/images/{self.ground_truth_filename}")
        self.ground_truth_image.setPixmap(ground_truth_pixmap)

        output_pixmap = QPixmap(f"algorithms/invertingGradients/images/{self.output_filename}")
        self.output_image.setPixmap(output_pixmap)

        self.elapsed_time_display.setText(f"Elapsed time: {self.elapsed_time}")
        self.reconstruct_loss_display.setText(f"Reconstruct loss: {self.rec_loss}")
from PyQt5.QtWidgets import (QWidget, QGridLayout, QSpinBox, QLabel, QHBoxLayout,
    QComboBox, QDoubleSpinBox)

class ParameterWidget(QWidget):
    def __init__(self):
        super().__init__()

        # self.setStyleSheet("color: white;"
        #                     "background-color: #496e9c;"
        #                     "border-style: solid;"
        #                     "border-width: 1px;"
        #                     "border-color: white;"
        #                     "border-radius: 1px")
        self.model = "ResNet20-4"
        self.dataset = "CIFAR10"
        self.cost_fn = "sim"
        self.indices = "def"
        self.restarts = 32
        self.target_id = -1

        grid = QGridLayout()  
        self.setLayout(grid)

        model_box = QHBoxLayout()
        model_input = QComboBox()
        model_input.addItems([
            "ConvNet", "ConvNet8", "ConvNet16", "ConvNet32",
            "BeyondInferringMNIST", "BeyondInferringCifar",
            "MLP", "TwoLP", "ResNet20", "ResNet20-nostride", "ResNet20-10",
            "ResNet20-4", "ResNet20-4-unpooled", "ResNet28-10",
            "ResNet32", "ResNet32-10", "ResNet44", "ResNet56", "ResNe110",
            "ResNet18", "ResNet34", "ResNet50", "ResNet50-2",
            "ResNet101", "ResNet152", "MobileNet", "MNASNet",
            "DenseNet121", "DenseNet40", "DenseNet40-4", 
            "SRNet3", "SRNet1", "iRevNet", "LetNetZhu"])
        model_input.currentTextChanged.connect(self.model_text_changed)
        model_label = QLabel("Model:")
        model_label.setBuddy(model_input)
        model_box.addWidget(model_label)
        model_box.addWidget(model_input)

        dataset_box = QHBoxLayout()
        dataset_input = QComboBox()
        dataset_input.addItems(["CIFAR10", "CIFAR100", "MNIST", "MNIST_GRAY", "ImageNet", "BSDS-SR", "BSDS-DN","BSDS-RGB"])
        dataset_input.currentTextChanged.connect(self.dataset_text_changed)
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
        restarts_input = QDoubleSpinBox()
        restarts_input.setValue(self.restarts)
        restarts_input.valueChanged.connect(lambda i: self.value_changed(i, "restarts"))
        restarts_label = QLabel("Restarts:")
        restarts_label.setBuddy(restarts_input)
        restarts_box.addWidget(restarts_label)
        restarts_box.addWidget(restarts_input)

        target_id_box = QHBoxLayout()
        target_id_input = QDoubleSpinBox()
        target_id_input.setMinimum(-10)
        target_id_input.setValue(self.target_id)
        target_id_input.valueChanged.connect(lambda i: self.value_changed(i, "target_id"))
        target_id_label = QLabel("Target_id:")
        target_id_label.setBuddy(target_id_input)
        target_id_box.addWidget(target_id_label)
        target_id_box.addWidget(target_id_input)

        grid.addLayout(model_box, 0, 0)
        grid.addLayout(dataset_box, 0, 1)
        grid.addLayout(cost_fn_box, 1, 0)
        grid.addLayout(indices_box, 1, 1)
        grid.addLayout(restarts_box, 2, 0)
        grid.addLayout(target_id_box, 2, 1)

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
from PyQt5.QtWidgets import (QWidget, QGridLayout, QSpinBox, QLabel, QHBoxLayout,
    QComboBox, QDoubleSpinBox)

class ParameterWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.num_of_clients = 3
        self.dataset = "mnist"
        self.learning_rate = 0.01
        self.alpha = 0.1
        self.total_epochs = 50

        grid = QGridLayout()  
        self.setLayout(grid)

        num_of_clients_box = QHBoxLayout()
        num_of_clients_input = QSpinBox()
        num_of_clients_input.setValue(self.num_of_clients)
        num_of_clients_input.valueChanged.connect(lambda i: self.value_changed(i, "num_of_clients"))
        num_of_clients_label = QLabel("Number of Clients:")
        num_of_clients_label.setBuddy(num_of_clients_input)
        num_of_clients_box.addWidget(num_of_clients_label)
        num_of_clients_box.addWidget(num_of_clients_input)

        dataset_box = QHBoxLayout()
        dataset_input = QComboBox()
        dataset_input.addItems(["MNIST", "EMINST"])
        dataset_input.currentTextChanged.connect(self.text_changed)
        dataset_label = QLabel("Dataset:")
        dataset_label.setBuddy(dataset_input)
        dataset_box.addWidget(dataset_label)
        dataset_box.addWidget(dataset_input)

        learning_rate_box = QHBoxLayout()
        learning_rate_input = QDoubleSpinBox()
        learning_rate_input.setValue(self.learning_rate)
        learning_rate_input.valueChanged.connect(lambda i: self.value_changed(i, "learning_rate"))
        learning_rate_label = QLabel("Learning Rate:")
        learning_rate_label.setBuddy(learning_rate_input)
        learning_rate_box.addWidget(learning_rate_label)
        learning_rate_box.addWidget(learning_rate_input)

        alpha_box = QHBoxLayout()
        alpha_input = QDoubleSpinBox()
        alpha_input.setValue(self.alpha)
        alpha_input.valueChanged.connect(lambda i: self.value_changed(i, "alpha"))
        alpha_label = QLabel("Alpha:")
        alpha_label.setBuddy(alpha_input)
        alpha_box.addWidget(alpha_label)
        alpha_box.addWidget(alpha_input)

        total_epochs_box = QHBoxLayout()
        total_epochs_input = QSpinBox()
        total_epochs_input.setValue(self.total_epochs)
        total_epochs_input.valueChanged.connect(lambda i: self.value_changed(i, "total_epochs"))
        total_epochs_label = QLabel("Total epochs:")
        total_epochs_label.setBuddy(total_epochs_input)
        total_epochs_box.addWidget(total_epochs_label)
        total_epochs_box.addWidget(total_epochs_input)

        grid.addLayout(num_of_clients_box, 0, 0)
        grid.addLayout(dataset_box, 0, 1)
        grid.addLayout(learning_rate_box, 1, 0)
        grid.addLayout(alpha_box, 1, 1)

    def value_changed(self, i, name):
        self.__dict__[name] = i

    def text_changed(self, s):
        if s == "MNIST":
            self.dataset = "Mnist"
        else:
            self.dataset = "EMnist"
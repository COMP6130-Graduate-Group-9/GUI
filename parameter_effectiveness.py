from PyQt5.QtWidgets import (QWidget, QGridLayout, QSpinBox, QLabel, QHBoxLayout,
    QComboBox)

class ParameterWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.num_of_clients = 3
        self.dataset = "mnist"

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

        grid.addLayout(num_of_clients_box, 0, 0)
        grid.addLayout(dataset_box, 0, 1)

    def value_changed(self, i, name):
        self.__dict__[name] = i

    def text_changed(self, s):
        if s == "MNIST":
            self.dataset = "Mnist"
        else:
            self.dataset = "EMnist"
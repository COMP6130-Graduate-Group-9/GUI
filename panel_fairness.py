from PyQt5.QtGui import QFont
from PyQt5.QtCore import QProcess, left, right
from PyQt5.QtWidgets import (QBoxLayout, QComboBox, QDialog, QGroupBox, QLabel, QProgressBar, QSpinBox, QStackedLayout, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QPlainTextEdit)

import numpy
import os

class PanelFairness(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, parent=parent)
        
        self.p = None   # Default empty value.
        
        titleLabel = QLabel("Fairness")
        titleFont = QFont()
        titleFont.setPointSize(24)
        titleLabel.setFont(titleFont)
        
        topLayout = QHBoxLayout()
        topLayout.addWidget(titleLabel)
        self.createChildren()
        
        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0)
        mainLayout.addLayout(self.panelDisplay, 1, 0)
        self.setLayout(mainLayout)
    
    def createChildren(self):
        self.panelDisplay = QStackedLayout()
        page1 = FairnessDataset(self)
        page2 = Setup(self)
        page3 = Experiment(self)
        
        self.panelDisplay.addWidget(page1)
        self.panelDisplay.addWidget(page2)
        self.panelDisplay.addWidget(page3)
        self.panelDisplay.setCurrentIndex(0)
    
    def switch_panel(self, index):
        self.panelDisplay.setCurrentIndex(index)
    
    
class FairnessDataset(PanelFairness):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, parent=parent)
        self.parent = parent  
        
        self.p = None     
        self.rounds = 0
        
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)        
        self.defaultFont = QFont()
        self.defaultFont.setPointSize(9)
        self.createDatasetButtons()
        self.createParameters()
        
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.btn_generate_dataset)
        leftLayout.addWidget(self.btn_run_experiment)
        
        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.text)
        
        toplayout = QHBoxLayout()
        toplayout.addLayout(leftLayout)
        toplayout.addLayout(self.grid)
        toplayout.addLayout(rightLayout)
        
        bottomLayout = QHBoxLayout()
        bottomLayout.addWidget(self.progressbarLabel)
        bottomLayout.addWidget(self.progressbar)
        
        mainLayout = QVBoxLayout()
        mainLayout.addLayout(toplayout)
        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)        
        
        #layout = QVBoxLayout()
        #layout.addWidget(self.btn_generate_dataset)
        #layout.addWidget(self.text)
        #layout.addWidget(self.btn_go_to_setup)
        #self.setLayout(layout)
        
    
    def createDatasetButtons(self):        
        self.btn_generate_dataset = QPushButton("Generate dataset")
        self.btn_generate_dataset.pressed.connect(self.generate_dataset)
        self.btn_run_experiment = QPushButton("Run Experiment")
        self.btn_run_experiment.pressed.connect(self.run_experiment)
        self.progressbarLabel = QLabel('Progress...')
        self.progressbarLabel.setFont(self.defaultFont)
        self.progressbar = QProgressBar(self)
    
    def createParameters(self):
        self.method = "fedavg"
        self.model = "cnn"
        self.numRounds = 20
        self.batchSize = 10
        
        self.grid = QGridLayout()
        
        methodLabel = QLabel("Method: ")
        methodLabel.setFont(self.defaultFont)        
        methodCombobox = QComboBox()
        methodCombobox.addItems(["fedavg", "fedfv", "fedprox"])
        methodCombobox.setFont(self.defaultFont)
        methodCombobox.currentTextChanged.connect(self.methodTextChanged)
        
        modelLabel = QLabel("Model: ")
        modelLabel.setFont(self.defaultFont)        
        modelCombobox = QComboBox()
        modelCombobox.addItems(["cnn", "mlp", "resnet18"])
        modelCombobox.setFont(self.defaultFont)
        modelCombobox.currentTextChanged.connect(self.modelTextChanged)
        
        numRoundsLabel = QLabel("Rounds: ")
        numRoundsLabel.setFont(self.defaultFont)
        numRoundsSpinBox = QSpinBox()
        numRoundsSpinBox.setFont(self.defaultFont)
        numRoundsSpinBox.setValue(self.numRounds)
        numRoundsSpinBox.valueChanged.connect(lambda i: self.value_changed(i, "numRounds"))
        
        batchsizeLabel = QLabel("Batch Size: ")
        batchsizeLabel.setFont(self.defaultFont)
        batchSizeSpinBox = QSpinBox()
        batchSizeSpinBox.setFont(self.defaultFont)
        batchSizeSpinBox.setValue(self.batchSize)
        batchSizeSpinBox.valueChanged.connect(lambda i: self.value_changed(i, "batchSize"))        
        
        self.grid.addWidget(methodLabel, 0, 0)
        self.grid.addWidget(methodCombobox, 0, 1)
        self.grid.addWidget(modelLabel, 1, 0)
        self.grid.addWidget(modelCombobox, 1, 1)
        self.grid.addWidget(numRoundsLabel, 2, 0)
        self.grid.addWidget(numRoundsSpinBox, 2, 1)
        self.grid.addWidget(batchsizeLabel, 3, 0)
        self.grid.addWidget(batchSizeSpinBox, 3, 1)        
        
    
    def generate_dataset(self):
        if self.p is None:  # No process running.
            self.message("Executing process")
            self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
            self.p.readyReadStandardOutput.connect(self.handle_stdout)
            self.p.readyReadStandardError.connect(self.handle_stderr)
            self.p.stateChanged.connect(self.handle_state)
            return_path = "../../"
            self.p.finished.connect(lambda: self.process_finished(return_path))  # Clean up once complete.
            os.chdir("algorithms/easyFL")
            script = "generate_fedtask.py"
            self.p.start(f"{return_path}venv/Scripts/python.exe", script.split(" "))
    
    def run_experiment(self):
        if self.p is None:  # No process running.
            self.message("Executing process")
            self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
            self.p.readyReadStandardOutput.connect(self.handle_stdout)
            self.p.readyReadStandardError.connect(self.handle_stderr)
            self.p.stateChanged.connect(self.handle_state)
            return_path = "../../"
            self.p.finished.connect(lambda: self.process_finished(return_path))  # Clean up once complete.
            os.chdir("algorithms/easyFL")
            #script = "main.py --task mnist_client100_dist0_beta0_noise0 --model cnn --method fedavg --num_rounds 20 --num_epochs 5 --learning_rate 0.215 --proportion 0.1 --batch_size 10 --train_rate 1 --eval_interval 1"
            script = "main.py --task mnist_client100_dist0_beta0_noise0 --model " + self.model + " --method " + self.method + " --num_rounds " + str(self.numRounds) + " --num_epochs 5 --learning_rate 0.215 --proportion 0.1 --batch_size " + str(self.batchSize) + " --train_rate 1 --eval_interval 1"
            self.p.start(f"{return_path}venv/Scripts/python.exe", script.split(" "))
    
    def message(self, s):
        self.text.appendPlainText(s)
        if (s.find('%') != -1):
            percentages = s.splitlines()
            for percent in percentages:
                if (percent.find('%') != -1):
                    percentage = float(percent[:len(percent)-1])
                    if (percentage <= 100):
                        self.progressbar.setValue(percentage) 
                    else:
                        self.progressbar.setValue(100)
        elif(s.find('init fedtask...') != -1):
            self.progressbar.setValue(0)
        elif (s.find('Round ') != -1):
            self.rounds = self.rounds + 1
            self.progressbar.setValue(self.rounds*(100/(self.numRounds + 2)))        
        elif (s.find('==End==') != -1):
            self.rounds = self.rounds + 1
            self.progressbar.setValue(self.rounds*(100/(self.numRounds + 2)))
            self.progressbar.setValue(100)            
        elif (s.find('Mean Time Cost of Each Round') != -1):
            self.progressbar.setValue(100)    

    def methodTextChanged(self, s):
        if s == "FEDFV":
            self.method = "fedfv"
        elif s == "FEDAVG":
            self.method = "fedavg"
        elif s == "FEDPROX":
            self.method = "fedprox"
        else:
            self.method = "fedavg"
            
    def modelTextChanged(self, s):
        if s == "MLP":
            self.model = "mlp"
        elif s == "CNN":
            self.model = "cnn"
        elif s == "FEDPROX":
            self.model = "RESNET18"
        else:
            self.model = "cnn"
    
    def value_changed(self, i, name):
        self.__dict__[name] = i
    
    def go_to_setup(self):
        if self.parent != None:
            self.parent.switch_panel(1)
    
    def message1(self, s):
        self.text.appendPlainText(s)     
    
    def handle_stderr(self):
        data = self.p.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        self.message(stderr)

    def handle_stdout(self):
        data = self.p.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        self.message(stdout)

    def handle_state(self, state):
        states = {
            QProcess.NotRunning: 'Not running',
            QProcess.Starting: 'Starting',
            QProcess.Running: 'Running',
        }
        state_name = states[state]
        self.message(f"State changed: {state_name}")
    
    def process_finished(self, return_path=None):
        self.message("Process finished.")
        self.p = None
        if return_path is not None:
            os.chdir(return_path)

class Setup(PanelFairness):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, parent=parent)
        self.parent = parent 
        
        self.p = None
        
        titleLabel = QLabel("Setup")
        titleFont = QFont()
        titleFont.setPointSize(24)
        titleLabel.setFont(titleFont)
        
        topLayout = QHBoxLayout()
        topLayout.addWidget(titleLabel)
        
        self.createSetup()
        
        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0)
        mainLayout.addLayout(self.grid, 1, 0)
        self.setLayout(mainLayout)
    
    def createSetup(self):
        self.grid = QGridLayout()
        methodLabel = QLabel("Method: ")
        methodLabelFont = QFont()
        methodLabelFont.setPointSize(9)
        methodLabel.setFont(methodLabelFont)        
        methodCombobox = QComboBox()
        methodCombobox.addItem("fedfv")
        methodCombobox.addItem("fedavg")
        methodCombobox.addItem("fedprox")
        methodCombobox.setFont(methodLabelFont)
        
        modelLabel = QLabel("Model: ")
        modelLabelFont = QFont()
        modelLabelFont.setPointSize(9)
        modelLabel.setFont(modelLabelFont)        
        modelCombobox = QComboBox()
        modelCombobox.addItem("mlp")
        modelCombobox.addItem("cnn")
        modelCombobox.addItem("resnet18")
        modelCombobox.setFont(modelLabelFont)
        
        
        self.grid.addWidget(methodLabel, 0, 0)
        self.grid.addWidget(methodCombobox, 0, 1)
        self.grid.addWidget(modelLabel, 1, 0)
        self.grid.addWidget(modelCombobox, 1, 1)
        
        self.btn_go_to_generate = QPushButton("Back")
        self.btn_go_to_generate.pressed.connect(self.go_to_generate)
        self.btn_run_experiment = QPushButton("Next")
        self.btn_run_experiment.pressed.connect(self.go_to_experiment) 
        
        
        self.grid.addWidget(self.btn_run_experiment, 1, 2)
        self.grid.addWidget(self.btn_go_to_generate, 2, 2)
    
    
    def go_to_generate(self):
        if self.parent != None:
            self.parent.switch_panel(0)
            
    def go_to_experiment(self):
        if self.parent != None:
            self.parent.switch_panel(2)

class Experiment(PanelFairness):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, parent=parent)
        self.parent = parent 
        
        self.p = None
        
        self.rounds = 0
        
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        
        titleLabel = QLabel("Experiment")
        titleFont = QFont()
        titleFont.setPointSize(24)
        titleLabel.setFont(titleFont)
        
        self.btn_run_experiment = QPushButton("Run")
        self.btn_run_experiment.pressed.connect(self.run_experiment)
        self.btn_go_to_setup = QPushButton("Back")
        self.btn_go_to_setup.pressed.connect(self.go_to_setup) 
        self.progressbar = QProgressBar(self)
        #self.progressbar.setGeometry(30, 40, 200, 25)
        
        topLayout = QHBoxLayout()
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(titleLabel)
        leftLayout.addWidget(self.btn_run_experiment)
        leftLayout.addWidget(self.btn_go_to_setup)
        leftLayout.addWidget(self.progressbar)
        #topLayout.addWidget(leftLayout)
        topLayout.addWidget(self.text)
        
        #self.createSetup()
        
        mainLayout = QGridLayout()
        mainLayout.addLayout(leftLayout, 0, 0)
        mainLayout.addLayout(topLayout, 0, 1)
        self.setLayout(mainLayout)
        ##self.run_experiment()
    
    def run_experiment(self):
        if self.p is None:  # No process running.
            self.message("Executing process")
            self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
            self.p.readyReadStandardOutput.connect(self.handle_stdout)
            self.p.readyReadStandardError.connect(self.handle_stderr)
            self.p.stateChanged.connect(self.handle_state)
            return_path = "../../"
            self.p.finished.connect(lambda: self.process_finished(return_path))  # Clean up once complete.
            os.chdir("algorithms/easyFL")
            script = "main.py --task mnist_client100_dist0_beta0_noise0 --model cnn --method fedavg --num_rounds 20 --num_epochs 5 --learning_rate 0.215 --proportion 0.1 --batch_size 10 --train_rate 1 --eval_interval 1"
            self.p.start(f"{return_path}venv/Scripts/python.exe", script.split(" "))
            
                
    def go_to_setup(self):
        if self.parent != None:
            self.parent.switch_panel(1)
    
    def message(self, s):
        self.text.appendPlainText(s)
        if (s.find('Round ') != -1):
            self.rounds = self.rounds + 1
            self.progressbar.setValue(self.rounds*(100/22))
        
        if (s.find('==End==') != -1):
            self.rounds = self.rounds + 1
            self.progressbar.setValue(self.rounds*(100/22))
            self.progressbar.setValue(100)
            
        if (s.find('Mean Time Cost of Each Round') != -1):
            self.progressbar.setValue(100)
    
    
    def handle_stderr(self):
        data = self.p.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        self.message(stderr)

    def handle_stdout(self):
        data = self.p.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        self.message(stdout)

    def handle_state(self, state):
        states = {
            QProcess.NotRunning: 'Not running',
            QProcess.Starting: 'Starting',
            QProcess.Running: 'Running',
        }
        state_name = states[state]
        self.message(f"State changed: {state_name}")
    
    def process_finished(self, return_path=None):
        self.message("Process finished.")
        self.p = None
        if return_path is not None:
            os.chdir(return_path)
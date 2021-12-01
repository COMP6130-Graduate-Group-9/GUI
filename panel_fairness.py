from PyQt5.QtGui import QFont
from PyQt5.QtCore import QProcess, left, right
from PyQt5.QtWidgets import (QBoxLayout, QComboBox, QDialog, QGroupBox, QLabel, QProgressBar, QSpinBox, QStackedLayout, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QPlainTextEdit)
from PyQt5 import QtCore

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pathlib import Path
from datetime import datetime
from fairness import result_analysis
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
        self.page1 = Experiment(self)
        self.page2 = Results(self)
        
        self.panelDisplay.addWidget(self.page1)
        self.panelDisplay.addWidget(self.page2)
        self.panelDisplay.setCurrentIndex(0)
    
    def switch_panel(self, index):
        self.panelDisplay.setCurrentIndex(index)
    
    
class Experiment(PanelFairness):
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
        leftLayout.addWidget(self.btn_show_results)
        
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
    
    def createDatasetButtons(self):        
        self.btn_generate_dataset = QPushButton("Generate dataset")
        self.btn_generate_dataset.pressed.connect(self.generate_dataset)
        self.btn_run_experiment = QPushButton("Run Experiment")
        self.btn_run_experiment.pressed.connect(self.run_experiment)
        self.btn_show_results = QPushButton("Show Results")
        self.btn_show_results.pressed.connect(self.show_results)
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
            if (str(Path(__file__).absolute()).find('utils') != -1):
                os.chdir("../../..")
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
            self.rounds = 0
            if (str(Path(__file__).absolute()).find('utils') != -1):
                os.chdir("../../..")
            self.message("Executing process")
            self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
            self.p.readyReadStandardOutput.connect(self.handle_stdout)
            self.p.readyReadStandardError.connect(self.handle_stderr)
            self.p.stateChanged.connect(self.handle_state)
            return_path = "../../"
            self.p.finished.connect(lambda: self.process_finished(return_path))  # Clean up once complete.
            os.chdir("algorithms/easyFL")
            script = "main.py --task mnist_client100_dist0_beta0_noise0 --model " + self.model + " --method " + self.method + " --num_rounds " + str(self.numRounds) + " --num_epochs 5 --learning_rate 0.215 --proportion 0.1 --batch_size " + str(self.batchSize) + " --train_rate 1 --eval_interval 1"
            self.p.start(f"{return_path}venv/Scripts/python.exe", script.split(" "))
    
    def show_results(self):      
        self.go_to_results()
        return
    
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
    
    def go_to_results(self):
        if self.parent != None:
            self.parent.page2.plot()
            self.parent.switch_panel(1)    
    
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

class Results(PanelFairness):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, parent=parent)
        self.parent = parent  
        
        self.p = None 
        
        self.btn_show_results = QPushButton("Show Results")
        self.btn_show_results.setFixedSize(QtCore.QSize(100, 20))
        self.btn_show_results.pressed.connect(self.plot)
        self.btn_save = QPushButton("Save Results")
        self.btn_save.setFixedSize(QtCore.QSize(100, 20))
        self.btn_save.pressed.connect(self.save_plot)
        self.btn_go_back = QPushButton("Back")
        self.btn_go_back.setFixedSize(QtCore.QSize(100, 20))
        self.btn_go_back.pressed.connect(self.go_back)        
        
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.btn_save)
        leftLayout.addWidget(self.btn_go_back)
        
        layout = QHBoxLayout()
        layout.addLayout(leftLayout)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def plot(self):
        self.figure.clear()        
        
        if (str(Path(__file__).absolute()).find('utils') != -1):
            os.chdir("../../..")
            
        os.chdir("algorithms/easyFL/utils")
        ax, x , y = result_analysis.create_graphs()
        
        task = 'mnist_client100_dist0_beta0_noise0'
        curve_names = [
            'train_losses',
            'test_losses',
            'test_accs',
        ]        
        
        # train losses
        ax1 = self.figure.add_subplot(311)
        ax1.plot(x[0], y[0], linewidth=1)
        plt.title(task + " (" + curve_names[0] + ")")
        plt.xlabel("communication rounds")
        plt.ylabel(curve_names[0])
        ax = plt.gca()
        plt.grid()
        
        # test losses
        ax2 = self.figure.add_subplot(312)
        ax2.plot(x[1], y[1], linewidth=1)
        plt.title(task + " (" + curve_names[1] + ")")
        plt.xlabel("communication rounds")
        plt.ylabel(curve_names[1])
        ax = plt.gca()
        plt.grid()
        
        # test accuracy
        ax3 = self.figure.add_subplot(313)
        ax3.plot(x[2], y[2], linewidth=1)
        plt.title(task + " (" + curve_names[2] + ")")
        plt.xlabel("communication rounds")
        plt.ylabel(curve_names[2])
        ax = plt.gca()
        plt.grid()  
              
        plt.tight_layout()        
        
        self.canvas.draw()        
        os.chdir("../../..") 
    
    def save_plot(self):
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M%S")
        plt.savefig(dt_string + "_fairness_results.jpg")    
    
    def go_back(self):
        if self.parent != None:
            self.parent.switch_panel(0)
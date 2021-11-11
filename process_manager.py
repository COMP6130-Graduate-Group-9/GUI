from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, QProcess

import os

class ProcessManager(QObject):
    started = pyqtSignal()
    finished = pyqtSignal()
    textChanged = pyqtSignal(str)

    def __init__(self, exec_dir, return_path, parent=None):
        super().__init__(parent)

        self._exec_dir = exec_dir
        self._return_path = return_path

        self._process = QProcess(self)
        self._process.readyReadStandardError.connect(self.onReadyReadStandardError)
        self._process.readyReadStandardOutput.connect(self.onReadyReadStandardOutput)
        self._process.stateChanged.connect(self.onStateChanged)
        self._process.started.connect(self.started)
        self._process.finished.connect(self.finished)
        self._process.finished.connect(self.onFinished)

    def run_script(self, script=""):
        os.chdir(self._exec_dir)
        self._process.start(f"{self._return_path}venv/Scripts/python.exe", script.split(" "))

    @pyqtSlot(QProcess.ProcessState)
    def onStateChanged(self, state):
        if state == QProcess.NotRunning:
            print("not running")
        elif state == QProcess.Starting:
            print("starting")
        elif state == QProcess.Running:
            print("running")

    @pyqtSlot(int, QProcess.ExitStatus)
    def onFinished(self, exitCode, exitStatus):
        print(exitCode, exitStatus)
        os.chdir(self._return_path)

    @pyqtSlot()
    def onReadyReadStandardError(self):
        message = self._process.readAllStandardError().data().decode()
        print("error:", message)
        # self.finished.emit()
        # self._process.kill()
        self.textChanged.emit(message)

    @pyqtSlot()
    def onReadyReadStandardOutput(self):
        message = self._process.readAllStandardOutput().data().decode()
        self.textChanged.emit(message)
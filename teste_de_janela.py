from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import sys

# Classe principal da janela
class MainWindow(QWidget):

    # Construtor da classe MainWindow
    def __init__(self):
        super().__init__()

        # Configurações da janela
        self.setWindowTitle("Teste de Janela")
        self.setGeometry(100, 100, 400, 300)
        self.centralizar()

    # Método para centralizar a janela na tela
    def centralizar(self):
        qReact = self.frameGeometry()
        centerPoint = QApplication.primaryScreen().availableGeometry().center()
        qReact.moveCenter(centerPoint)
        self.move(qReact.topLeft())


# Início do programa
meuApp = QApplication(sys.argv)
window = MainWindow()
window.show()

meuApp.exec()
sys.exit()



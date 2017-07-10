import sys

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QMainWindow, QApplication, QAction, QPushButton, QLineEdit


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import scipy.misc

class Categorizer(QMainWindow):
    g = ""

    def __init__(self, parent=None):
        super(Categorizer,self).__init__(parent)

        self.initUI()

    def initUI(self):

        self.m = PlotCanvas(self, width=5, height=4)
        self.m.move(0,0)

        self.textbox = QLineEdit(self)
        self.textbox.move(20, 400)
        self.textbox.resize(280,25)

        btn = QPushButton('Next', self)
        btn.resize(btn.sizeHint())
        btn.move(330, 400)
        btn.clicked.connect(self.plotDefect)

        self.show()

    def plotDefect(self):
        self.g = self.g + self.textbox.text()
        self.textbox.setText("")
        self.m.plot()

class PlotCanvas(FigureCanvas):
    i=0
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        self.plot()


    def plot(self):
        INPUTFILE = "train/" + str(self.i) + ".tif"
        SEM = scipy.misc.imread(INPUTFILE,flatten=True)
        ax = self.figure.add_subplot(111)
        ax.imshow(SEM, cmap='gray')
        # ax.set_title('PyQt Matplotlib Example')
        self.i = self.i+1
        self.draw()

def main():
    app = QApplication(sys.argv)
    form = Categorizer()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()


import os
from os import walk
from os import listdir
from os.path import isfile, join
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit,QFileDialog
from PyQt5.QtGui import QIcon, QImage
from PyQt5.QtGui import QPixmap
import ThisIsML 

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 496)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 0, 181, 571))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.groupBox.setFont(font)
        self.groupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox.setObjectName("groupBox")
        self.openfolder = QtWidgets.QPushButton(self.groupBox)
        self.openfolder.setGeometry(QtCore.QRect(10, 50, 161, 61))
        self.openfolder.setObjectName("openfolder")
        self.nextimage = QtWidgets.QPushButton(self.groupBox)
        self.nextimage.setGeometry(QtCore.QRect(10, 130, 161, 61))
        self.nextimage.setObjectName("nextimage")
        self.preimage = QtWidgets.QPushButton(self.groupBox)
        self.preimage.setGeometry(QtCore.QRect(10, 210, 161, 61))
        self.preimage.setObjectName("preimage")
        self.save = QtWidgets.QPushButton(self.groupBox)
        self.save.setGeometry(QtCore.QRect(10, 290, 161, 61))
        self.save.setObjectName("save")
        self.detect = QtWidgets.QPushButton(self.groupBox)
        self.detect.setGeometry(QtCore.QRect(10, 360, 161, 61))
        self.detect.setObjectName("detect")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(610, 0, 181, 571))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_2.setObjectName("groupBox_2")
        self.selectmodel = QtWidgets.QPushButton(self.groupBox_2)
        self.selectmodel.setGeometry(QtCore.QRect(0, 30, 171, 41))
        self.selectmodel.setObjectName("selectmodel")
        
        self.r1 = QtWidgets.QRadioButton(self.groupBox_2)
        self.r1.setGeometry(QtCore.QRect(10, 90, 161, 31))
        self.r1.setObjectName("r1")
        
        self.r2 = QtWidgets.QRadioButton(self.groupBox_2)
        self.r2.setGeometry(QtCore.QRect(10, 130, 161, 31))
        self.r2.setObjectName("r2")
        
        self.r3 = QtWidgets.QRadioButton(self.groupBox_2)
        self.r3.setGeometry(QtCore.QRect(10, 170, 161, 31))
        self.r3.setObjectName("r3")
        
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(10, 210, 161, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox.setGeometry(QtCore.QRect(30, 240, 86, 25))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        
        self.person = QtWidgets.QCheckBox(self.groupBox_2)
        self.person.setGeometry(QtCore.QRect(20, 310, 151, 41))
        self.person.setObjectName("person")
        self.person.stateChanged.connect(lambda: self.clickBox(self.person))
        
        self.car = QtWidgets.QCheckBox(self.groupBox_2)
        self.car.setGeometry(QtCore.QRect(20, 360, 151, 41))
        self.car.setObjectName("car")
        self.car.stateChanged.connect(lambda: self.clickBox(self.car))
        
        self.airplane = QtWidgets.QCheckBox(self.groupBox_2)
        self.airplane.setGeometry(QtCore.QRect(20, 410, 151, 41))
        self.airplane.setObjectName("airplane")
        self.airplane.stateChanged.connect(lambda: self.clickBox(self.airplane))
        
        self.dog = QtWidgets.QCheckBox(self.groupBox_2)
        self.dog.setGeometry(QtCore.QRect(20, 460, 151, 41))
        self.dog.setObjectName("dog")
        self.dog.stateChanged.connect(lambda: self.clickBox(self.dog))
        
        self.apple = QtWidgets.QCheckBox(self.groupBox_2)
        self.apple.setGeometry(QtCore.QRect(20, 510, 151, 41))
        self.apple.setObjectName("apple")
        self.apple.stateChanged.connect(lambda: self.clickBox(self.apple))
        
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(16, 270, 151, 41))
        self.label_3.setObjectName("label_3")
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(200, 60, 411, 351))
        self.image.setObjectName("image")
        
        self.image2 = QtWidgets.QLabel(self.centralwidget)
        self.image2.setGeometry(QtCore.QRect(200, 400, 411, 351))
        self.image2.setObjectName("image2")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.model = 0
        self.index=-1
        self.images=[]
        self.size = -1
        self.path=""
        self.toDetectOnlyThese = []
        self.pixmap = QPixmap()
        
        self.openfolder.clicked.connect(self.open_folder)   
        self.detect.clicked.connect(self.RunTheCode)
        self.r1.clicked.connect(self.R1)
        self.r2.clicked.connect(self.R2)
        self.r3.clicked.connect(self.R3)
        self.nextimage.clicked.connect(self.NextImg)
        self.preimage.clicked.connect(self.PretImg)
        #self.save.clicked.connect(self.SAVE)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    def clickBox(self,b):
        if b.isChecked() == True:
            self.toDetectOnlyThese.append(b.text())
        else :
            self.toDetectOnlyThese.remove(b.text())
        print(self.toDetectOnlyThese)
    
    def NextImg(self):
        print(self.index)
        self.path = self.images[self.index+1]
        pixmap = QPixmap(self.path)
        self.image.setPixmap(pixmap)
        self.image.setScaledContents(True)
        self.index+=1
        
    def PretImg(self):
        print(self.index)
        self.path = self.images[self.index-1]
        pixmap = QPixmap(self.path)
        self.image.setPixmap(pixmap)
        self.image.setScaledContents(True)
        self.index-=1
        
    def R1(self):
        self.model = 0
    def R2(self):
        self.model = 1
    def R3(self):
        self.model = 2
        
        
    def RunTheCode(self):
        detection_score = float(self.comboBox.currentText())
        #Now we have path and the model, Make use of self.model also
        print(self.path)
        print(self.model)
        self.pixmap = QPixmap((self.path))
        image = ThisIsML.GiveMePathsReturnsListOfImg(self.path, self.toDetectOnlyThese ,self.model ,detection_score=detection_score)
        image = QtGui.QImage(image, image.shape[1],image.shape[0], image.shape[1] * 3,QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap(image)
        print("Function executed:")
        self.image2.setPixmap(pix)
        self.image2.setScaledContents(True)
        
    def open_folder(self):
        file = QFileDialog.getExistingDirectory()
        mypath = str(file)
        print(mypath)
        self.images = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
        self.size = len(self.images)
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.openfolder.setText(_translate("MainWindow", "Open Folder"))
        self.nextimage.setText(_translate("MainWindow", "Next Image"))
        self.preimage.setText(_translate("MainWindow", "Pre Image"))
        self.save.setText(_translate("MainWindow", "Save"))
        self.detect.setText(_translate("MainWindow", "Detect"))
        self.groupBox_2.setTitle(_translate("MainWindow", "GroupBox"))
        self.selectmodel.setText(_translate("MainWindow", "Select Model"))
        
        self.r1.setText(_translate("MainWindow", "SSDLiteMOBILENET"))####   CHANGE THIS TEXT
        self.r2.setText(_translate("MainWindow", "SSDMOBILENET18"))
        self.r3.setText(_translate("MainWindow", "FRCNN"))
        
        self.label_2.setText(_translate("MainWindow", "Detection Thershold"))
        self.comboBox.setItemText(0, _translate("MainWindow", "0"))
        self.comboBox.setItemText(1, _translate("MainWindow", "0.05"))
        self.comboBox.setItemText(2, _translate("MainWindow", "0.1"))
        self.comboBox.setItemText(3, _translate("MainWindow", "0.15"))
        self.comboBox.setItemText(4, _translate("MainWindow", "0.2"))
        self.comboBox.setItemText(5, _translate("MainWindow", "0.25"))
        self.comboBox.setItemText(6, _translate("MainWindow", "0.3"))
        self.comboBox.setItemText(7, _translate("MainWindow", "0.35"))
        self.comboBox.setItemText(8, _translate("MainWindow", "0.4"))
        self.comboBox.setItemText(9, _translate("MainWindow", "0.45"))
        self.comboBox.setItemText(10, _translate("MainWindow", "0.5"))
        self.comboBox.setItemText(11, _translate("MainWindow", "0.55"))
        self.comboBox.setItemText(12, _translate("MainWindow", "0.6"))
        self.comboBox.setItemText(13, _translate("MainWindow", "0.65"))
        self.comboBox.setItemText(14, _translate("MainWindow", "0.7"))
        self.comboBox.setItemText(15, _translate("MainWindow", "0.75"))
        self.comboBox.setItemText(16, _translate("MainWindow", "0.8"))
        self.comboBox.setItemText(17, _translate("MainWindow", "0.85"))
        self.comboBox.setItemText(18, _translate("MainWindow", "0.9"))
        self.comboBox.setItemText(19, _translate("MainWindow", "0.95"))
        self.comboBox.setItemText(20, _translate("MainWindow", "1"))
        
        self.person.setText(_translate("MainWindow", "person"))
        self.apple.setText(_translate("MainWindow", "apple"))
        self.airplane.setText(_translate("MainWindow", "airplane"))
        self.car.setText(_translate("MainWindow", "car"))
        self.dog.setText(_translate("MainWindow", "dog"))      
        
        self.label_3.setText(_translate("MainWindow", "Label Filter"))
        self.image.setText(_translate("MainWindow", "TextLabel"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

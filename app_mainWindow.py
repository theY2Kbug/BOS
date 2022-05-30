# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'app.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1080, 644)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1080, 640))
        MainWindow.setMaximumSize(QtCore.QSize(1080, 640))
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(1080, 600))
        self.centralwidget.setMaximumSize(QtCore.QSize(1080, 600))
        self.centralwidget.setObjectName("centralwidget")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(0, 0, 1081, 591))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.stackedWidget.setFont(font)
        self.stackedWidget.setStyleSheet("background:#fff;")
        self.stackedWidget.setObjectName("stackedWidget")
        self.home = QtWidgets.QWidget()
        self.home.setObjectName("home")
        self.home_video_feed = QtWidgets.QLabel(self.home)
        self.home_video_feed.setGeometry(QtCore.QRect(10, 10, 640, 480))
        self.home_video_feed.setMinimumSize(QtCore.QSize(640, 480))
        self.home_video_feed.setStyleSheet("border: 2px solid black")
        self.home_video_feed.setText("")
        self.home_video_feed.setAlignment(QtCore.Qt.AlignCenter)
        self.home_video_feed.setObjectName("home_video_feed")
        self.lower_hue = QtWidgets.QSlider(self.home)
        self.lower_hue.setGeometry(QtCore.QRect(660, 50, 331, 41))
        self.lower_hue.setMaximum(179)
        self.lower_hue.setTracking(True)
        self.lower_hue.setOrientation(QtCore.Qt.Horizontal)
        self.lower_hue.setObjectName("lower_hue")
        self.hue_lower_label = QtWidgets.QLabel(self.home)
        self.hue_lower_label.setGeometry(QtCore.QRect(660, 30, 131, 21))
        self.hue_lower_label.setObjectName("hue_lower_label")
        self.upper_hue = QtWidgets.QSlider(self.home)
        self.upper_hue.setGeometry(QtCore.QRect(660, 120, 331, 41))
        self.upper_hue.setMaximum(179)
        self.upper_hue.setSliderPosition(25)
        self.upper_hue.setTracking(True)
        self.upper_hue.setOrientation(QtCore.Qt.Horizontal)
        self.upper_hue.setObjectName("upper_hue")
        self.hue_upper_label = QtWidgets.QLabel(self.home)
        self.hue_upper_label.setGeometry(QtCore.QRect(660, 100, 131, 21))
        self.hue_upper_label.setObjectName("hue_upper_label")
        self.lower_sat = QtWidgets.QSlider(self.home)
        self.lower_sat.setGeometry(QtCore.QRect(660, 210, 331, 41))
        self.lower_sat.setMaximum(255)
        self.lower_sat.setSliderPosition(40)
        self.lower_sat.setTracking(True)
        self.lower_sat.setOrientation(QtCore.Qt.Horizontal)
        self.lower_sat.setObjectName("lower_sat")
        self.sat_lower_label = QtWidgets.QLabel(self.home)
        self.sat_lower_label.setGeometry(QtCore.QRect(660, 190, 181, 21))
        self.sat_lower_label.setObjectName("sat_lower_label")
        self.upper_sat = QtWidgets.QSlider(self.home)
        self.upper_sat.setGeometry(QtCore.QRect(660, 280, 331, 41))
        self.upper_sat.setMaximum(255)
        self.upper_sat.setSliderPosition(255)
        self.upper_sat.setTracking(True)
        self.upper_sat.setOrientation(QtCore.Qt.Horizontal)
        self.upper_sat.setObjectName("upper_sat")
        self.sat_upper_label = QtWidgets.QLabel(self.home)
        self.sat_upper_label.setGeometry(QtCore.QRect(660, 260, 181, 21))
        self.sat_upper_label.setObjectName("sat_upper_label")
        self.lower_val = QtWidgets.QSlider(self.home)
        self.lower_val.setGeometry(QtCore.QRect(660, 370, 331, 41))
        self.lower_val.setMaximum(255)
        self.lower_val.setSliderPosition(20)
        self.lower_val.setTracking(True)
        self.lower_val.setOrientation(QtCore.Qt.Horizontal)
        self.lower_val.setObjectName("lower_val")
        self.val_lower_label = QtWidgets.QLabel(self.home)
        self.val_lower_label.setGeometry(QtCore.QRect(660, 350, 141, 21))
        self.val_lower_label.setObjectName("val_lower_label")
        self.upper_val = QtWidgets.QSlider(self.home)
        self.upper_val.setGeometry(QtCore.QRect(660, 440, 331, 41))
        self.upper_val.setMaximum(255)
        self.upper_val.setSliderPosition(220)
        self.upper_val.setTracking(True)
        self.upper_val.setOrientation(QtCore.Qt.Horizontal)
        self.upper_val.setObjectName("upper_val")
        self.val_upper_label = QtWidgets.QLabel(self.home)
        self.val_upper_label.setGeometry(QtCore.QRect(660, 420, 141, 21))
        self.val_upper_label.setObjectName("val_upper_label")
        self.lh = QtWidgets.QLabel(self.home)
        self.lh.setGeometry(QtCore.QRect(1000, 30, 67, 51))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.lh.setFont(font)
        self.lh.setStyleSheet("border:1px solid #000000;\n"
"border-radius:10px;")
        self.lh.setAlignment(QtCore.Qt.AlignCenter)
        self.lh.setObjectName("lh")
        self.uh = QtWidgets.QLabel(self.home)
        self.uh.setGeometry(QtCore.QRect(1000, 100, 67, 51))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.uh.setFont(font)
        self.uh.setStyleSheet("border:1px solid #000000;\n"
"border-radius:10px;")
        self.uh.setAlignment(QtCore.Qt.AlignCenter)
        self.uh.setObjectName("uh")
        self.ls = QtWidgets.QLabel(self.home)
        self.ls.setGeometry(QtCore.QRect(1000, 190, 67, 51))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.ls.setFont(font)
        self.ls.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.ls.setStyleSheet("border:1px solid #000000;\n"
"border-radius:10px;")
        self.ls.setAlignment(QtCore.Qt.AlignCenter)
        self.ls.setObjectName("ls")
        self.us = QtWidgets.QLabel(self.home)
        self.us.setGeometry(QtCore.QRect(1000, 260, 67, 51))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.us.setFont(font)
        self.us.setStyleSheet("border:1px solid #000000;\n"
"border-radius:10px;")
        self.us.setAlignment(QtCore.Qt.AlignCenter)
        self.us.setObjectName("us")
        self.lv = QtWidgets.QLabel(self.home)
        self.lv.setGeometry(QtCore.QRect(1000, 350, 67, 51))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.lv.setFont(font)
        self.lv.setStyleSheet("border:1px solid #000000;\n"
"border-radius:10px;")
        self.lv.setAlignment(QtCore.Qt.AlignCenter)
        self.lv.setObjectName("lv")
        self.uv = QtWidgets.QLabel(self.home)
        self.uv.setGeometry(QtCore.QRect(1000, 420, 67, 51))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.uv.setFont(font)
        self.uv.setStyleSheet("border:1px solid #000000;\n"
"border-radius:10px;")
        self.uv.setAlignment(QtCore.Qt.AlignCenter)
        self.uv.setObjectName("uv")
        self.line = QtWidgets.QFrame(self.home)
        self.line.setGeometry(QtCore.QRect(660, 155, 411, 31))
        self.line.setStyleSheet("color: #000;")
        self.line.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line.setLineWidth(2)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.home)
        self.line_2.setGeometry(QtCore.QRect(660, 320, 411, 21))
        self.line_2.setStyleSheet("color: #000;")
        self.line_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_2.setLineWidth(2)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setObjectName("line_2")
        self.next_btn = QtWidgets.QCommandLinkButton(self.home)
        self.next_btn.setGeometry(QtCore.QRect(290, 520, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.next_btn.setFont(font)
        self.next_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.next_btn.setStyleSheet("QCommandLinkButton{\n"
"    border:1px solid blue;\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QCommandLinkButton::hover{\n"
"    background-color:rgb(166, 202, 240);\n"
"}")
        self.next_btn.setIconSize(QtCore.QSize(20, 20))
        self.next_btn.setObjectName("next_btn")
        self.start_feed_btn = QtWidgets.QPushButton(self.home)
        self.start_feed_btn.setGeometry(QtCore.QRect(20, 510, 91, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.start_feed_btn.setFont(font)
        self.start_feed_btn.setStyleSheet("QPushButton{\n"
"    color:#fff;\n"
"    background: rgb(78, 154, 6);\n"
"    border:1px solid grey;\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QPushButton::hover\n"
"{\n"
"    background-color : #fff;\n"
"    color:rgb(40, 108, 5);\n"
"}")
        self.start_feed_btn.setObjectName("start_feed_btn")
        self.setHSV = QtWidgets.QPushButton(self.home)
        self.setHSV.setGeometry(QtCore.QRect(820, 510, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.setHSV.setFont(font)
        self.setHSV.setStyleSheet("QPushButton::hover\n"
"{\n"
"    background-color : #000;\n"
"    color:#fff;\n"
"}")
        self.setHSV.setObjectName("setHSV")
        self.fps = QtWidgets.QGroupBox(self.home)
        self.fps.setGeometry(QtCore.QRect(130, 500, 141, 61))
        self.fps.setObjectName("fps")
        self.rb15 = QtWidgets.QRadioButton(self.fps)
        self.rb15.setGeometry(QtCore.QRect(10, 30, 41, 21))
        self.rb15.setObjectName("rb15")
        self.rb30 = QtWidgets.QRadioButton(self.fps)
        self.rb30.setGeometry(QtCore.QRect(80, 30, 41, 21))
        self.rb30.setChecked(True)
        self.rb30.setObjectName("rb30")
        self.stackedWidget.addWidget(self.home)
        self.last = QtWidgets.QWidget()
        self.last.setObjectName("last")
        self.video_feed = QtWidgets.QLabel(self.last)
        self.video_feed.setGeometry(QtCore.QRect(10, 10, 848, 480))
        self.video_feed.setStyleSheet("border:2px solid;")
        self.video_feed.setText("")
        self.video_feed.setObjectName("video_feed")
        self.detect_btn = QtWidgets.QPushButton(self.last)
        self.detect_btn.setGeometry(QtCore.QRect(870, 20, 181, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.detect_btn.setFont(font)
        self.detect_btn.setStyleSheet("QPushButton\n"
"{\n"
"    background: #fff;\n"
"    border:1px solid grey;\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QPushButton::hover\n"
"{\n"
"    background-color:rgb(186, 189, 182);\n"
"}")
        self.detect_btn.setObjectName("detect_btn")
        self.segment_btn = QtWidgets.QPushButton(self.last)
        self.segment_btn.setGeometry(QtCore.QRect(870, 100, 181, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.segment_btn.setFont(font)
        self.segment_btn.setStyleSheet("QPushButton\n"
"{\n"
"    background: #fff;\n"
"    border:1px solid grey;\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QPushButton::hover\n"
"{\n"
"    background-color:rgb(186, 189, 182);\n"
"}")
        self.segment_btn.setObjectName("segment_btn")
        self.threed_view_btn = QtWidgets.QPushButton(self.last)
        self.threed_view_btn.setGeometry(QtCore.QRect(870, 180, 181, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.threed_view_btn.setFont(font)
        self.threed_view_btn.setStyleSheet("QPushButton\n"
"{\n"
"    background: #fff;\n"
"    border:1px solid grey;\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QPushButton::hover\n"
"{\n"
"    background-color:rgb(186, 189, 182);\n"
"}")
        self.threed_view_btn.setObjectName("threed_view_btn")
        self.bos_btn = QtWidgets.QPushButton(self.last)
        self.bos_btn.setGeometry(QtCore.QRect(870, 260, 181, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.bos_btn.setFont(font)
        self.bos_btn.setStyleSheet("QPushButton\n"
"{\n"
"    background: #fff;\n"
"    border:1px solid grey;\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QPushButton::hover\n"
"{\n"
"    background-color:rgb(186, 189, 182);\n"
"}")
        self.bos_btn.setObjectName("bos_btn")
        self.stop_record_btn = QtWidgets.QPushButton(self.last)
        self.stop_record_btn.setGeometry(QtCore.QRect(870, 420, 81, 61))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.stop_record_btn.setFont(font)
        self.stop_record_btn.setStyleSheet("QPushButton\n"
"{\n"
"    color: #fff;\n"
"    background: #D70040;\n"
"    border:1px solid grey;\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QPushButton::hover\n"
"{\n"
"    background-color:#fff;\n"
"    color:#D70040;\n"
"}\n"
"\n"
"\n"
"")
        self.stop_record_btn.setObjectName("stop_record_btn")
        self.reset_btn = QtWidgets.QPushButton(self.last)
        self.reset_btn.setGeometry(QtCore.QRect(970, 420, 81, 61))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.reset_btn.setFont(font)
        self.reset_btn.setStyleSheet("QPushButton{\n"
"    color:#fff;\n"
"    background: #EE4B2B;\n"
"    border:1px solid grey;\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QPushButton::hover\n"
"{\n"
"    background-color : #fff;\n"
"    color:#EE4B2B;\n"
"}")
        self.reset_btn.setObjectName("reset_btn")
        self.home_btn = QtWidgets.QCommandLinkButton(self.last)
        self.home_btn.setGeometry(QtCore.QRect(10, 510, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.home_btn.setFont(font)
        self.home_btn.setStyleSheet("QCommandLinkButton{\n"
"border:1px solid blue;\n"
"border-radius: 10px;\n"
"}\n"
"\n"
"QCommandLinkButton::hover{\n"
"    background-color:rgb(166, 202, 240);\n"
"}\n"
"")
        self.home_btn.setObjectName("home_btn")
        self.detection_accuracy = QtWidgets.QSlider(self.last)
        self.detection_accuracy.setGeometry(QtCore.QRect(170, 540, 271, 21))
        self.detection_accuracy.setMinimum(1)
        self.detection_accuracy.setMaximum(99)
        self.detection_accuracy.setSliderPosition(40)
        self.detection_accuracy.setOrientation(QtCore.Qt.Horizontal)
        self.detection_accuracy.setObjectName("detection_accuracy")
        self.det_acc = QtWidgets.QLabel(self.last)
        self.det_acc.setGeometry(QtCore.QRect(450, 520, 41, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.det_acc.setFont(font)
        self.det_acc.setStyleSheet("border:2px solid black;")
        self.det_acc.setAlignment(QtCore.Qt.AlignCenter)
        self.det_acc.setObjectName("det_acc")
        self.detection_accuracy_label = QtWidgets.QLabel(self.last)
        self.detection_accuracy_label.setGeometry(QtCore.QRect(170, 510, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.detection_accuracy_label.setFont(font)
        self.detection_accuracy_label.setObjectName("detection_accuracy_label")
        self.groupBox = QtWidgets.QGroupBox(self.last)
        self.groupBox.setGeometry(QtCore.QRect(870, 320, 181, 91))
        self.groupBox.setObjectName("groupBox")
        self.record_screen_btn = QtWidgets.QPushButton(self.groupBox)
        self.record_screen_btn.setGeometry(QtCore.QRect(10, 30, 71, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.record_screen_btn.setFont(font)
        self.record_screen_btn.setStyleSheet("QPushButton{\n"
"    color:#fff;\n"
"    background: rgb(78, 154, 6);\n"
"    border:1px solid grey;\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QPushButton::hover\n"
"{\n"
"    background-color : #fff;\n"
"    color:rgb(40, 108, 5);\n"
"}")
        self.record_screen_btn.setObjectName("record_screen_btn")
        self.record_screen = QtWidgets.QCheckBox(self.groupBox)
        self.record_screen.setGeometry(QtCore.QRect(100, 30, 71, 23))
        self.record_screen.setObjectName("record_screen")
        self.record_3d = QtWidgets.QCheckBox(self.groupBox)
        self.record_3d.setGeometry(QtCore.QRect(100, 60, 41, 23))
        self.record_3d.setObjectName("record_3d")
        self.stackedWidget.addWidget(self.last)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1080, 22))
        self.menuBar.setObjectName("menuBar")
        self.menuAbout = QtWidgets.QMenu(self.menuBar)
        self.menuAbout.setObjectName("menuAbout")
        self.menuSettings = QtWidgets.QMenu(self.menuBar)
        self.menuSettings.setObjectName("menuSettings")
        MainWindow.setMenuBar(self.menuBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setStyleSheet("")
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.actionChange_Neural_Network = QtWidgets.QAction(MainWindow)
        self.actionChange_Neural_Network.setObjectName("actionChange_Neural_Network")
        self.actionImage_processing = QtWidgets.QAction(MainWindow)
        self.actionImage_processing.setObjectName("actionImage_processing")
        self.actionGithub = QtWidgets.QAction(MainWindow)
        self.actionGithub.setObjectName("actionGithub")
        self.actionHelp = QtWidgets.QAction(MainWindow)
        self.actionHelp.setObjectName("actionHelp")
        self.actionSave_Location = QtWidgets.QAction(MainWindow)
        self.actionSave_Location.setObjectName("actionSave_Location")
        self.menuAbout.addAction(self.actionGithub)
        self.menuAbout.addAction(self.actionHelp)
        self.menuSettings.addAction(self.actionImage_processing)
        self.menuSettings.addAction(self.actionSave_Location)
        self.menuBar.addAction(self.menuAbout.menuAction())
        self.menuBar.addAction(self.menuSettings.menuAction())

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Base of Support Estimator"))
        self.hue_lower_label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Hue </span>(Lower Limit)</p></body></html>"))
        self.hue_upper_label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Hue </span>(Upper Limit)</p></body></html>"))
        self.sat_lower_label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Saturation </span>(Lower Limit)</p></body></html>"))
        self.sat_upper_label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Saturation </span>(Upper Limit)</p></body></html>"))
        self.val_lower_label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Value </span>(Lower Limit)</p></body></html>"))
        self.val_upper_label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Value </span>(Upper Limit)</p></body></html>"))
        self.lh.setText(_translate("MainWindow", "0"))
        self.uh.setText(_translate("MainWindow", "25"))
        self.ls.setText(_translate("MainWindow", "40"))
        self.us.setText(_translate("MainWindow", "255"))
        self.lv.setText(_translate("MainWindow", "20"))
        self.uv.setText(_translate("MainWindow", "220"))
        self.next_btn.setText(_translate("MainWindow", "Next"))
        self.start_feed_btn.setText(_translate("MainWindow", "Start\n"
"Feed"))
        self.setHSV.setText(_translate("MainWindow", "Set"))
        self.fps.setTitle(_translate("MainWindow", "Frames Per Second"))
        self.rb15.setText(_translate("MainWindow", "15"))
        self.rb30.setText(_translate("MainWindow", "30"))
        self.detect_btn.setText(_translate("MainWindow", "Detect"))
        self.segment_btn.setText(_translate("MainWindow", "Segment"))
        self.threed_view_btn.setText(_translate("MainWindow", "3D View"))
        self.bos_btn.setText(_translate("MainWindow", "BOS"))
        self.stop_record_btn.setText(_translate("MainWindow", "Stop\n"
"Recording"))
        self.reset_btn.setText(_translate("MainWindow", "RESET"))
        self.home_btn.setText(_translate("MainWindow", "Home"))
        self.det_acc.setText(_translate("MainWindow", "40"))
        self.detection_accuracy_label.setText(_translate("MainWindow", "Detection Accuracy"))
        self.groupBox.setTitle(_translate("MainWindow", "Record"))
        self.record_screen_btn.setText(_translate("MainWindow", "Record"))
        self.record_screen.setText(_translate("MainWindow", "Screen"))
        self.record_3d.setText(_translate("MainWindow", "3D"))
        self.menuAbout.setTitle(_translate("MainWindow", "About"))
        self.menuSettings.setTitle(_translate("MainWindow", "Settings"))
        self.actionChange_Neural_Network.setText(_translate("MainWindow", "Change Neural Network"))
        self.actionImage_processing.setText(_translate("MainWindow", "Image Processing"))
        self.actionGithub.setText(_translate("MainWindow", "Github"))
        self.actionHelp.setText(_translate("MainWindow", "Help"))
        self.actionHelp.setShortcut(_translate("MainWindow", "F1"))
        self.actionSave_Location.setText(_translate("MainWindow", "Save Location"))

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'settings.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ImageSettingsWindow(object):
    def setupUi(self, ImageSettingsWindow):
        ImageSettingsWindow.setObjectName("ImageSettingsWindow")
        ImageSettingsWindow.resize(560, 300)
        ImageSettingsWindow.setMinimumSize(QtCore.QSize(560, 300))
        ImageSettingsWindow.setMaximumSize(QtCore.QSize(560, 300))
        self.centralwidget = QtWidgets.QWidget(ImageSettingsWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(290, 20, 241, 71))
        self.groupBox.setObjectName("groupBox")
        self.gaussian_blur_label = QtWidgets.QLabel(self.groupBox)
        self.gaussian_blur_label.setGeometry(QtCore.QRect(200, 30, 31, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.gaussian_blur_label.setFont(font)
        self.gaussian_blur_label.setStyleSheet("border:1px solid black;\n"
"background-color:#fff;")
        self.gaussian_blur_label.setAlignment(QtCore.Qt.AlignCenter)
        self.gaussian_blur_label.setObjectName("gaussian_blur_label")
        self.gaussian_blur = QtWidgets.QSlider(self.groupBox)
        self.gaussian_blur.setGeometry(QtCore.QRect(10, 40, 181, 16))
        self.gaussian_blur.setMinimum(1)
        self.gaussian_blur.setMaximum(7)
        self.gaussian_blur.setSingleStep(2)
        self.gaussian_blur.setProperty("value", 3)
        self.gaussian_blur.setOrientation(QtCore.Qt.Horizontal)
        self.gaussian_blur.setObjectName("gaussian_blur")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 110, 241, 71))
        self.groupBox_2.setObjectName("groupBox_2")
        self.decimation_label = QtWidgets.QLabel(self.groupBox_2)
        self.decimation_label.setGeometry(QtCore.QRect(200, 30, 31, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.decimation_label.setFont(font)
        self.decimation_label.setStyleSheet("border:1px solid black;\n"
"background-color:#fff;")
        self.decimation_label.setAlignment(QtCore.Qt.AlignCenter)
        self.decimation_label.setObjectName("decimation_label")
        self.decimation_threshold = QtWidgets.QSlider(self.groupBox_2)
        self.decimation_threshold.setGeometry(QtCore.QRect(10, 40, 181, 16))
        self.decimation_threshold.setMinimum(2)
        self.decimation_threshold.setMaximum(4)
        self.decimation_threshold.setOrientation(QtCore.Qt.Horizontal)
        self.decimation_threshold.setObjectName("decimation_threshold")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(290, 110, 241, 71))
        self.groupBox_3.setObjectName("groupBox_3")
        self.iou_label = QtWidgets.QLabel(self.groupBox_3)
        self.iou_label.setGeometry(QtCore.QRect(200, 30, 31, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.iou_label.setFont(font)
        self.iou_label.setStyleSheet("border:1px solid black;\n"
"background-color:#fff;")
        self.iou_label.setAlignment(QtCore.Qt.AlignCenter)
        self.iou_label.setObjectName("iou_label")
        self.iou_threshold = QtWidgets.QSlider(self.groupBox_3)
        self.iou_threshold.setGeometry(QtCore.QRect(10, 40, 181, 16))
        self.iou_threshold.setMinimum(1)
        self.iou_threshold.setSliderPosition(45)
        self.iou_threshold.setOrientation(QtCore.Qt.Horizontal)
        self.iou_threshold.setObjectName("iou_threshold")
        self.depth_enable = QtWidgets.QCheckBox(self.centralwidget)
        self.depth_enable.setGeometry(QtCore.QRect(200, 40, 71, 51))
        self.depth_enable.setObjectName("depth_enable")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(20, 200, 241, 71))
        self.groupBox_4.setObjectName("groupBox_4")
        self.contour_label = QtWidgets.QLabel(self.groupBox_4)
        self.contour_label.setGeometry(QtCore.QRect(190, 30, 41, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.contour_label.setFont(font)
        self.contour_label.setStyleSheet("border:1px solid black;\n"
"background-color:#fff;")
        self.contour_label.setAlignment(QtCore.Qt.AlignCenter)
        self.contour_label.setObjectName("contour_label")
        self.contour_threshold = QtWidgets.QSlider(self.groupBox_4)
        self.contour_threshold.setGeometry(QtCore.QRect(10, 40, 171, 16))
        self.contour_threshold.setMinimum(0)
        self.contour_threshold.setMaximum(9999)
        self.contour_threshold.setSliderPosition(500)
        self.contour_threshold.setOrientation(QtCore.Qt.Horizontal)
        self.contour_threshold.setObjectName("contour_threshold")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(290, 200, 241, 71))
        self.groupBox_5.setObjectName("groupBox_5")
        self.opening_kernel = QtWidgets.QSlider(self.groupBox_5)
        self.opening_kernel.setGeometry(QtCore.QRect(10, 40, 181, 16))
        self.opening_kernel.setMinimum(1)
        self.opening_kernel.setMaximum(7)
        self.opening_kernel.setSingleStep(2)
        self.opening_kernel.setSliderPosition(3)
        self.opening_kernel.setOrientation(QtCore.Qt.Horizontal)
        self.opening_kernel.setObjectName("opening_kernel")
        self.opening_kernel_label = QtWidgets.QLabel(self.groupBox_5)
        self.opening_kernel_label.setGeometry(QtCore.QRect(200, 30, 31, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.opening_kernel_label.setFont(font)
        self.opening_kernel_label.setStyleSheet("border:1px solid black;\n"
"background-color:#fff;")
        self.opening_kernel_label.setAlignment(QtCore.Qt.AlignCenter)
        self.opening_kernel_label.setObjectName("opening_kernel_label")
        self.segmentation = QtWidgets.QGroupBox(self.centralwidget)
        self.segmentation.setGeometry(QtCore.QRect(20, 30, 161, 61))
        self.segmentation.setObjectName("segmentation")
        self.radio_hsv = QtWidgets.QRadioButton(self.segmentation)
        self.radio_hsv.setGeometry(QtCore.QRect(10, 30, 51, 23))
        self.radio_hsv.setChecked(True)
        self.radio_hsv.setObjectName("radio_hsv")
        self.radio_regular = QtWidgets.QRadioButton(self.segmentation)
        self.radio_regular.setGeometry(QtCore.QRect(70, 30, 81, 23))
        self.radio_regular.setObjectName("radio_regular")
        ImageSettingsWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(ImageSettingsWindow)
        QtCore.QMetaObject.connectSlotsByName(ImageSettingsWindow)

    def retranslateUi(self, ImageSettingsWindow):
        _translate = QtCore.QCoreApplication.translate
        ImageSettingsWindow.setWindowTitle(_translate("ImageSettingsWindow", "Image Settings"))
        self.groupBox.setTitle(_translate("ImageSettingsWindow", "Gaussian Blur Window Size"))
        self.gaussian_blur_label.setText(_translate("ImageSettingsWindow", "3"))
        self.groupBox_2.setTitle(_translate("ImageSettingsWindow", "Decimation Filter Magnitude"))
        self.decimation_label.setText(_translate("ImageSettingsWindow", "2"))
        self.groupBox_3.setTitle(_translate("ImageSettingsWindow", "IoU Threshold"))
        self.iou_label.setText(_translate("ImageSettingsWindow", "45"))
        self.depth_enable.setText(_translate("ImageSettingsWindow", "Depth\n"
"View"))
        self.groupBox_4.setTitle(_translate("ImageSettingsWindow", "Contour Area Threshold"))
        self.contour_label.setText(_translate("ImageSettingsWindow", "500"))
        self.groupBox_5.setTitle(_translate("ImageSettingsWindow", "Opening Kernel Size"))
        self.opening_kernel_label.setText(_translate("ImageSettingsWindow", "3"))
        self.segmentation.setTitle(_translate("ImageSettingsWindow", "Segmentation Type"))
        self.radio_hsv.setText(_translate("ImageSettingsWindow", "HSV"))
        self.radio_regular.setText(_translate("ImageSettingsWindow", "Normal"))
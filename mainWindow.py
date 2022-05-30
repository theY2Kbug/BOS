import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyrealsense2 as rs
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tflite_runtime.interpreter as tflite
from tensorflow import shape,reshape,concat,constant,cast,split,boolean_mask,float32
from tensorflow import image as image
from tensorflow import math as math
import random
import colorsys
import webbrowser
import datetime

# from matplotlib.figure import Figure
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from app_mainWindow import Ui_MainWindow
from app_settingsWindow import Ui_ImageSettingsWindow
from app_helpWindow import Ui_HelpWindow


# print(tf.test.is_built_with_cuda())
record = 0
fps = 30
lowerHue = 0
upperHue = 25
lowerSat = 40
upperSat = 255
lowerVal = 20
upperVal = 220
saveDir = os.getcwd()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = 1

contour_settings = 500
iou_threshold = 45
decimation_filter_mag = 2
gaussian_blur = 3
hsv_enabled = True
regular_enabled = False
opening_kernel_size = 3
depthFlag = False

class mainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_win)
        self.settings_ui = Ui_ImageSettingsWindow()
        self.ui.stackedWidget.setCurrentWidget(self.ui.home)

        self.ui.actionGithub.triggered.connect(lambda: webbrowser.open('https://github.com/theY2Kbug/BOS#base-of-support-bos'))
        self.ui.actionSave_Location.triggered.connect(self.openSaveFileLocation)
        # self.ui.actionHelp.triggered.connect(lambda: self.ui.openHelp())
        self.ui.next_btn.setEnabled(False)
        self.ui.setHSV.setEnabled(False)
        self.ui.lower_hue.setEnabled(False)
        self.ui.upper_hue.setEnabled(False)
        self.ui.lower_sat.setEnabled(False)
        self.ui.upper_sat.setEnabled(False)
        self.ui.lower_val.setEnabled(False)
        self.ui.upper_val.setEnabled(False)
        self.ui.threed_view_btn.setEnabled(False)
        self.ui.next_btn.clicked.connect(self.showLast)
        self.ui.home_btn.clicked.connect(self.showHome)
        self.ui.setHSV.clicked.connect(self.enableLast)
        self.ui.lower_hue.valueChanged.connect(self.setLowerHue)
        self.ui.upper_hue.valueChanged.connect(self.setUpperHue)
        self.ui.lower_sat.valueChanged.connect(self.setLowerSat)
        self.ui.upper_sat.valueChanged.connect(self.setUpperSat)
        self.ui.lower_val.valueChanged.connect(self.setLowerVal)
        self.ui.upper_val.valueChanged.connect(self.setUpperVal)
        self.ui.detect_btn.clicked.connect(self.detect)
        self.ui.reset_btn.clicked.connect(self.reset)
        self.ui.segment_btn.clicked.connect(self.segment)
        self.ui.bos_btn.clicked.connect(self.bos)
        self.ui.detection_accuracy.valueChanged.connect(self.changeDetectionThresh)
        self.ui.start_feed_btn.clicked.connect(self.startCam)
        self.ui.record_screen.toggled.connect(self.setRecord)
        self.ui.record_screen_btn.setEnabled(False)
        self.ui.record_screen_btn.clicked.connect(self.record)
        self.ui.stop_record_btn.clicked.connect(self.stop_record)
        self.ui.record_3d.setEnabled(False)
        self.ui.actionHelp.triggered.connect(lambda: self.openHelp())
        self.ui.actionImage_processing.setEnabled(False)
        self.ui.actionImage_processing.triggered.connect(lambda: self.openSettings())
        self.ui.rb15.toggled.connect(self.setFPS15)
        self.ui.rb30.toggled.connect(self.setFPS30)
        self.ui.statusBar.showMessage(f"Set FPS and start feed")
        # self.ui.next_btn.clicked.connect(self.stopCam)

        # self.settings_ui.depth_enable.setEnabled(False)

    def record(self):
        global record
        global fps
        global saveDir
        self.ui.stop_record_btn.setEnabled(True)
        record = 1
        global fourcc,out
        ct = datetime.datetime.now()
        out = cv2.VideoWriter(f"{saveDir}/{ct}.avi", fourcc, 6, (848, 480))

    def stop_record(self):
        global record
        global out
        out.release()
        self.ui.record_screen.setChecked(False)
        self.ui.record_screen_btn.setEnabled(False)
        self.ui.stop_record_btn.setEnabled(False)
        record = 0

    def bos(self):
        self.thread_1.bos_flag = True

    def setFPS15 (self):
        global fps
        fps = 15

    def setFPS30 (self):
        global fps
        fps = 30

    def openHelp(self):
        self.window = QMainWindow()
        self.help_ui = Ui_HelpWindow()
        self.help_ui.setupUi(self.window)
        self.window.show()

    def openSettings(self):
        global contour_settings
        global decimation_filter_mag
        global opening_kernel_size
        global iou_threshold
        global gaussian_blur
        global depthFlag
        self.window = QMainWindow()
        self.settings_ui = Ui_ImageSettingsWindow()
        self.settings_ui.setupUi(self.window)
        self.window.show()

        if(hsv_enabled):
            self.settings_ui.radio_hsv.setChecked(True)
        elif(regular_enabled):
            self.settings_ui.radio_regular.setChecked(True)
        self.settings_ui.radio_hsv.toggled.connect(self.setHSVflag)
        self.settings_ui.radio_regular.toggled.connect(self.setRegularflag)

        if(depthFlag):
            self.settings_ui.depth_enable.setChecked(True)
        else:
            self.settings_ui.depth_enable.setChecked(False)
        self.settings_ui.depth_enable.toggled.connect(self.enableDepth)

        self.settings_ui.contour_threshold.setSliderPosition(contour_settings)
        self.settings_ui.contour_label.setText(str(contour_settings))
        self.settings_ui.contour_threshold.valueChanged.connect(self.setContourThreshold)

        self.settings_ui.decimation_threshold.setSliderPosition(decimation_filter_mag)
        self.settings_ui.decimation_label.setText(str(decimation_filter_mag))
        self.settings_ui.decimation_threshold.valueChanged.connect(self.setDecimationThreshold)

        self.settings_ui.gaussian_blur.setSliderPosition(gaussian_blur)
        self.settings_ui.gaussian_blur_label.setText(str(gaussian_blur))
        self.settings_ui.gaussian_blur.valueChanged.connect(self.setGaussian)

        self.settings_ui.opening_kernel.setSliderPosition(opening_kernel_size)
        self.settings_ui.opening_kernel_label.setText(str(opening_kernel_size))
        self.settings_ui.opening_kernel.valueChanged.connect(self.setOpening)

        self.settings_ui.iou_threshold.setSliderPosition(iou_threshold)
        self.settings_ui.iou_label.setText(str(iou_threshold))
        self.settings_ui.iou_threshold.valueChanged.connect(self.setIOU)
        
    def enableDepth(self,enabled):
        global depthFlag
        if(enabled):
            depthFlag = True
        else:
            depthFlag = False

    def setContourThreshold(self,value):
        global contour_settings
        contour_settings = value
        self.settings_ui.contour_label.setText(str(value))

    def setOpening(self,value):
        global opening_kernel_size
        opening_kernel_size = value
        self.settings_ui.opening_kernel_label.setText(str(value))
    
    def setDecimationThreshold(self,value):
        global decimation_filter_mag
        decimation_filter_mag = value
        self.settings_ui.decimation_label.setText(str(value))

    def setGaussian(self,value):
        global gaussian_blur
        gaussian_blur = value
        self.settings_ui.gaussian_blur_label.setText(str(value))

    def setIOU(self,value):
        global iou_threshold
        iou_threshold = value
        self.settings_ui.iou_label.setText(str(value))

    def setRecord(self,enabled):
        if enabled:
            self.ui.record_screen_btn.setEnabled(True)
        else:
            self.ui.record_screen_btn.setEnabled(False)

    def setHSVflag(self,enabled):
        global hsv_enabled
        global regular_enabled
        if enabled:
            hsv_enabled = True
            regular_enabled = False
            self.thread_1.HSV_flag = True
            self.thread_1.Regular_flag = False

    def setRegularflag(self,enabled):
        global regular_enabled
        global hsv_enabled
        if enabled:
            regular_enabled = True
            hsv_enabled = False
            self.thread_1.HSV_flag = False
            self.thread_1.Regular_flag = True

    def show(self):
        self.main_win.show()

    def enableLast(self):
        global lowerHue,upperHue,lowerSat,lowerVal,upperSat,upperVal
        self.ui.next_btn.setEnabled(True)
        self.ui.statusBar.clearMessage()
        self.ui.statusBar.showMessage(f"Hue: {lowerHue}-{upperHue}; Saturation: {lowerSat} - {upperSat}; Value: {lowerVal} - {upperVal}, Click Next")

    def openSaveFileLocation(self):
        global saveDir
        file = str(QFileDialog.getExistingDirectory(None, "Select directory to save recording"))
        if(file != ''):
            saveDir = file
        self.ui.statusBar.clearMessage()
        self.ui.statusBar.showMessage(f"Click Next{saveDir}",2000)

    def showLast(self):
        self.ui.start_feed_btn.setEnabled(True)
        self.ui.stackedWidget.setCurrentWidget(self.ui.last)
        self.ui.bos_btn.setEnabled(False)
        self.ui.threed_view_btn.setEnabled(False)
        self.ui.stop_record_btn.setEnabled(False)
        self.ui.segment_btn.setEnabled(False)
        self.ui.detection_accuracy.setEnabled(False)
        self.ui.reset_btn.setEnabled(False)
        self.thread_0.stopThread()
        self.thread_1 = Camera(self.ui,width=848,height=480)
        self.thread_1.start()
        self.thread_1.img_signal.connect(self.ImageUpdateSlot_last)
        self.ui.statusBar.clearMessage()
        self.ui.statusBar.showMessage(f"Click Detect")

    def detect(self):
        self.thread_1.detect_flag = True
        self.ui.detection_accuracy.setEnabled(True)
        self.ui.segment_btn.setEnabled(True)
        self.ui.reset_btn.setEnabled(True)
        self.ui.statusBar.clearMessage()
        self.ui.statusBar.showMessage(f"Click Segment or 3D View")

    def segment(self):
        self.thread_1.segment_flag = True
        self.ui.threed_view_btn.setEnabled(False)
        self.ui.bos_btn.setEnabled(True)
        
    def changeDetectionThresh(self,value):
        self.thread_1.detect = value/100
        self.ui.det_acc.setText(str(value))

    def reset(self):
        self.thread_1.detect_flag = False
        self.thread_1.segment_flag = False
        self.thread_1.bos_flag = False
        self.ui.segment_btn.setEnabled(False)
        self.ui.reset_btn.setEnabled(False)
        self.ui.bos_btn.setEnabled(False)
        self.ui.detection_accuracy.setEnabled(False)

    def showHome(self):
        self.thread_1.stopThread()
        self.ui.stackedWidget.setCurrentWidget(self.ui.home)
        self.ui.setHSV.setEnabled(False)
        self.ui.next_btn.setEnabled(False)
        self.ui.lower_hue.setEnabled(False)
        self.ui.upper_hue.setEnabled(False)
        self.ui.lower_sat.setEnabled(False)
        self.ui.upper_sat.setEnabled(False)
        self.ui.lower_val.setEnabled(False)
        self.ui.upper_val.setEnabled(False)
        self.ui.actionImage_processing.setEnabled(False)
        self.ui.fps.setEnabled(True)
        self.ui.statusBar.clearMessage()
        self.ui.statusBar.showMessage(f"Set FPS and start feed")
        

    def setLowerHue(self,value):
        global lowerHue
        global upperHue
        if (value<=upperHue):
            lowerHue = value
            self.ui.lh.setText(str(value))

    def setUpperHue(self,value):
        global lowerHue
        global upperHue
        if (value>=lowerHue):
            upperHue = value
            self.ui.uh.setText(str(value))

    def setLowerSat(self,value):
        global lowerSat
        global upperSat
        if (value<=upperSat):
            lowerSat = value
            self.ui.ls.setText(str(value))

    def setUpperSat(self,value):
        global lowerSat
        global upperSat
        if (value>=lowerSat):
            upperSat = value
            self.ui.us.setText(str(value))

    def setLowerVal(self,value):
        global lowerVal
        global upperVal
        if (value<=upperVal):
            lowerVal = value
            self.ui.lv.setText(str(value))

    def setUpperVal(self,value):
        global lowerVal
        global upperVal
        if (value>=lowerVal):
            upperVal = value
            self.ui.uv.setText(str(value))

    def startCam(self):
        self.ui.setHSV.setEnabled(True)
        self.thread_0 = HomeCamera(self.ui,width=640,height=480)
        self.thread_0.start()
        self.thread_0.img_signal.connect(self.ImageUpdateSlot_home)
        self.ui.start_feed_btn.setEnabled(False)
        self.ui.lower_hue.setEnabled(True)
        self.ui.upper_hue.setEnabled(True)
        self.ui.lower_sat.setEnabled(True)
        self.ui.upper_sat.setEnabled(True)
        self.ui.lower_val.setEnabled(True)
        self.ui.upper_val.setEnabled(True)
        self.ui.actionImage_processing.setEnabled(True)
        self.ui.fps.setEnabled(False)
        self.ui.statusBar.clearMessage()
        self.ui.statusBar.showMessage(f"Adjust HSV to segment feet and click set")

    def ImageUpdateSlot_home(self,Image):
        self.ui.home_video_feed.setPixmap(QPixmap.fromImage(Image))

    def ImageUpdateSlot_last(self,Image):
        self.ui.video_feed.setPixmap(QPixmap.fromImage(Image))


class HomeCamera(QThread):
    img_signal = pyqtSignal(QImage)

    def __init__(self,main_win,parent=None,width=640,height=480):
        super(HomeCamera,self).__init__(parent)
        self.main_win = main_win
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()

    def run(self):
        global fps
        self.ThreadActive = True
        try:
            config = rs.config()
            config.enable_stream(rs.stream.color,self.width, self.height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)
            # print(fps)
            self.profile = self.pipeline.start(config)
            while self.ThreadActive:
                frame = self.pipeline.wait_for_frames()
                align = rs.align(rs.stream.color)
                aligned_frame = align.process(frame)
                depth_frame = aligned_frame.get_depth_frame()
                color_frame = aligned_frame.get_color_frame()
                color_im = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                if(color_im.shape[0]):
                    color_image = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)
                    hsv_image = cv2.cvtColor(color_im, cv2.COLOR_BGR2HSV)
                    lower = np.array([lowerHue, lowerSat, lowerVal])
                    upper = np.array([upperHue, upperSat, upperVal])
                    mask = cv2.inRange(hsv_image, lower, upper)
                    result = cv2.bitwise_and(color_image, color_image, mask=mask)
                    converttoQTformat = QImage(result.data,result.shape[1],result.shape[0],QImage.Format_RGB888)
                    self.img_signal.emit(converttoQTformat)
            
        except Exception as e:
            self.main_win.statusBar.clearMessage()
            self.main_win.statusBar.showMessage(f"Camera not detected, retry")
            self.main_win.start_feed_btn.setEnabled(True)
            self.main_win.fps.setEnabled(True)
            self.main_win.lower_hue.setEnabled(False)
            self.main_win.upper_hue.setEnabled(False)
            self.main_win.lower_sat.setEnabled(False)
            self.main_win.upper_sat.setEnabled(False)
            self.main_win.lower_val.setEnabled(False)
            self.main_win.upper_val.setEnabled(False)
            self.main_win.setHSV.setEnabled(False)
            self.main_win.actionImage_processing.setEnabled(False)
            pass

    def stopThread(self):
        self.ThreadActive = False
        self.pipeline.stop()
        self.quit()

class Camera(QThread):
    img_signal = pyqtSignal(QImage)
    def __init__(self,main_win,parent=None,width=848,height=480):
        global hsv_enabled
        global regular_enabled
        super(Camera,self).__init__(parent)
        self.main_win = main_win
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)
        self.profile = self.pipeline.start(self.config)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self.decimation = rs.decimation_filter()
        self.colorizer = rs.colorizer()
        self.interpreter = tflite.Interpreter(model_path="./model/yolov4-tiny-color-832_fp16.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.detect_flag = False
        self.segment_flag = False
        self.bos_flag = False
        self.detect = 0.4
        self.HSV_flag = hsv_enabled
        self.Regular_flag = regular_enabled
        

    def run(self):
        global fps
        global depthFlag
        global decimation_filter_mag
        global out
        global record
        self.ThreadActive = True
        
        while self.ThreadActive:
            frame = self.pipeline.wait_for_frames()
            align = rs.align(rs.stream.color)
            aligned_frame = align.process(frame)
            depth_frame = aligned_frame.get_depth_frame()
            color_frame = aligned_frame.get_color_frame()
            color_im = np.asanyarray(color_frame.get_data())
            self.decimation.set_option(rs.option.filter_magnitude, decimation_filter_mag)
            decimated_depth_frame = self.decimation.process(depth_frame)
            depth_image_decimated = np.asanyarray(decimated_depth_frame.get_data())
            colorized_depth = np.asanyarray(self.colorizer.colorize(decimated_depth_frame).get_data())
            if(color_im.shape[0]):
                self.color_image = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)
                if(self.detect_flag):
                    image_data = cv2.resize(self.color_image, (416,416))
                    image_data = image_data / 255.
                    image_data = image_data[np.newaxis, ...].astype(np.float32)
                    self.interpreter.set_tensor(self.input_details[0]['index'], image_data)
                    self.interpreter.invoke()
                    pred = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]
                    boxes, pred_conf = self.filter_boxes(pred[0], pred[1], score_threshold=self.detect, input_shape=constant([416, 416]))

                    boxes, scores, classes, valid_detections = image.combined_non_max_suppression(
                        boxes=reshape(boxes, (shape(boxes)[0], -1, 1, 4)),
                        scores=reshape(
                            pred_conf, (shape(pred_conf)[0], -1, shape(pred_conf)[-1])),
                        max_output_size_per_class=5,
                        max_total_size=10,
                        iou_threshold=0.45,
                        score_threshold=self.detect
                    )
                    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
                    if(self.segment_flag == False):
                        detected_image,detected_depth_image = self.draw_bbox(self.color_image,colorized_depth, pred_bbox)
                        if(depthFlag):
                            detected_image[0:detected_depth_image.shape[0],0:detected_depth_image.shape[1],:] = detected_depth_image
                        if(record):
                            temp = cv2.cvtColor(detected_image,cv2.COLOR_RGB2BGR)
                            out.write(temp)
                        imagetoQTformat = QImage(detected_image.data,detected_image.shape[1],detected_image.shape[0],QImage.Format_RGB888)
                        self.img_signal.emit(imagetoQTformat)
                    else:
                        segmented_image,contours = self.segment(self.color_image,pred_bbox)
                        if(self.bos_flag):
                            bos = self.draw_bos(contours)
                            if(record):
                                temp = cv2.cvtColor(bos,cv2.COLOR_RGB2BGR)
                                out.write(temp)
                            imagetoQTformat = QImage(bos.data,bos.shape[1],bos.shape[0],QImage.Format_RGB888)
                            self.img_signal.emit(imagetoQTformat)
                        else:
                            if(record):
                                temp = cv2.cvtColor(segmented_image,cv2.COLOR_RGB2BGR)
                                out.write(temp)
                            imagetoQTformat = QImage(segmented_image.data,segmented_image.shape[1],segmented_image.shape[0],QImage.Format_RGB888)
                            self.img_signal.emit(imagetoQTformat)
                else:
                    if(depthFlag):
                        self.color_image[0:colorized_depth.shape[0],0:colorized_depth.shape[1],:] = colorized_depth
                    if(record):
                        temp = cv2.cvtColor(self.color_image,cv2.COLOR_RGB2BGR)
                        out.write(temp)
                    converttoQTformat = QImage(self.color_image.data,self.color_image.shape[1],self.color_image.shape[0],QImage.Format_RGB888)
                    self.img_signal.emit(converttoQTformat)

    def draw_bbox(self,col, depth, bboxes, classes={0: 'RF', 1: 'LF', 2: 'RA', 3: 'LA', 4: 'RT', 5: 'LT'}):
        num_classes = len(classes)
        image_h, image_w, _ = col.shape
        depth_h,depth_w,_ = depth.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        out_boxes, out_scores, out_classes, num_boxes = bboxes
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
            coor = out_boxes[0][i]
            im_coor = np.copy(coor)
            depth_coor = np.copy(coor)
            im_coor[0] = int(im_coor[0] * image_h)
            im_coor[2] = int(im_coor[2] * image_h)
            im_coor[1] = int(im_coor[1] * image_w)
            im_coor[3] = int(im_coor[3] * image_w)

            depth_coor[0] = int(depth_coor[0] * depth_h)
            depth_coor[2] = int(depth_coor[2] * depth_h)
            depth_coor[1] = int(depth_coor[1] * depth_w)
            depth_coor[3] = int(depth_coor[3] * depth_w)

            score = out_scores[0][i]
            class_ind = int(out_classes[0][i])
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            if (classes[class_ind] == 'RF' or classes[class_ind] == 'LF'):
                c1, c2 = (im_coor[1], im_coor[0]), (im_coor[3], im_coor[2])
                depth_c1, depth_c2 = (depth_coor[1], depth_coor[0]), (depth_coor[3], depth_coor[2])
                cv2.rectangle(col, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), bbox_color, bbox_thick)
                cv2.rectangle(depth, (int(depth_c1[0]), int(depth_c1[1])), (int(depth_c2[0]), int(depth_c2[1])), bbox_color, bbox_thick)

        return col,depth

    def filter_boxes(self,box_xywh, scores, score_threshold=0.4, input_shape = constant([416,416])):
        scores_max = math.reduce_max(scores, axis=-1)
        mask = scores_max >= score_threshold
        class_boxes = boolean_mask(box_xywh, mask)
        pred_conf = boolean_mask(scores, mask)
        class_boxes = reshape(class_boxes, [shape(scores)[0], -1, shape(class_boxes)[-1]])
        pred_conf = reshape(pred_conf, [shape(scores)[0], -1, shape(pred_conf)[-1]])

        box_xy, box_wh = split(class_boxes, (2, 2), axis=-1)

        input_shape = cast(input_shape, dtype=float32)

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        box_mins = (box_yx - (box_hw / 2.)) / input_shape
        box_maxes = (box_yx + (box_hw / 2.)) / input_shape
        boxes = concat([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ], axis=-1)
        return (boxes, pred_conf)

    def segment(self,color_image,bboxes, classes={0: 'RF', 1: 'LF', 2: 'RA', 3: 'LA', 4: 'RT', 5: 'LT'},image_h = 480,image_w = 848):
        global contour_settings
        num_classes = len(classes)
        white_im = np.full((480, 848, 3),255, dtype = np.uint8)
        hsv_image = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2HSV)
        lower = np.array([lowerHue, lowerSat, lowerVal])
        upper = np.array([upperHue, upperSat, upperVal])
        mask = cv2.inRange(hsv_image, lower, upper)
        result = cv2.bitwise_and(color_image, color_image, mask=mask)
        blur = cv2.GaussianBlur(result,(5,5),0)
        toBeGrayed = np.copy(blur)
        if(self.Regular_flag):
            toBeGrayed = np.copy(self.color_image)        
        gray_image = cv2.cvtColor(toBeGrayed, cv2.COLOR_RGB2GRAY)
        ret3,thresh = cv2.threshold(gray_image,70,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if (self.HSV_flag):
            thresh = 255 - thresh
        dilate = cv2.dilate(thresh,None, iterations=1)
        out_boxes, out_scores, out_classes, num_boxes = bboxes
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
            coor = out_boxes[0][i]
            coor[0] = int(coor[0] * image_h)
            coor[2] = int(coor[2] * image_h)
            coor[1] = int(coor[1] * image_w)
            coor[3] = int(coor[3] * image_w)
            class_ind = int(out_classes[0][i])
            if (classes[class_ind] == 'RF' or classes[class_ind] == 'LF'):
                c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
                white_im[c1[1]:c2[1],c1[0]:c2[0],0] = dilate[c1[1]:c2[1],c1[0]:c2[0]]
        
        kernel = np.ones((3,3),np.uint8)
        open = cv2.morphologyEx(white_im[:,:,0], cv2.MORPH_OPEN, kernel)
        contours, hierarchy = cv2.findContours(image=(255-open), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        final_contours = []
        for contour in contours:
            if(cv2.contourArea(contour)<contour_settings):
                continue
            else:
                final_contours.append(contour)
        cv2.drawContours(image=white_im, contours=final_contours, contourIdx=-1, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)
        self.main_win.statusBar.clearMessage()
        self.main_win.statusBar.showMessage(f"No.of contours detected:{len(final_contours)}, proceed if contours are drawn only on feet")
        return white_im,final_contours

    def draw_bos(self,contours):
        final_metrics = []
        mask = np.copy(self.color_image)
        if(len(contours) == 1):
            bos = np.vstack(contours[0]).squeeze()
            cv2.fillPoly(mask,[bos],color = (255, 36, 0))
            bos_img = cv2.addWeighted(mask, 0.3,self.color_image,0.7, 0)
            return(bos_img)

        elif(len(contours) == 2):
            for contour in contours:
                cnt = np.vstack(contour).squeeze()
                if cnt.ndim != 2:
                    self.main_win.statusBar.clearMessage()
                    self.main_win.statusBar.showMessage(f"Numpy ndim error")
                    continue
                y_min_index = np.argmin(cnt[:,1])
                xy_minimum = tuple(cnt[y_min_index,:])
                # print(xy_minimum)
                y_max_index = np.argmax(cnt[:,1])
                xy_maximum = tuple(cnt[y_max_index,:])
                # print(xy_max)
                (x,y),radius = cv2.minEnclosingCircle(cnt)
                center = (int(x),int(y))
                final_metrics.append(dict({'center':center , 'xy_min':xy_minimum, 'xy_min_idx':y_min_index, 'xy_max':xy_maximum, 'xy_max_idx':y_max_index, 'points':cnt}))
            if(final_metrics[0]['center'][0]<final_metrics[1]['center'][0]):
                left = final_metrics[0]
                right = final_metrics[1]
            else:
                left = final_metrics[1]
                right = final_metrics[0]

            left_points = left['points'][left['xy_min_idx']:left['xy_max_idx']+1]
            right_points = right['points'][right['xy_max_idx']:]
            right_points = np.concatenate([right_points, right['points'][0:right['xy_min_idx']]])
            bos = np.concatenate([left_points,right_points])
            cv2.fillPoly(mask,[bos],color = (255, 36, 0))
            bos_img = cv2.addWeighted(mask, 0.3,self.color_image,0.7, 0)
            return(bos_img)

        else:
            self.main_win.statusBar.clearMessage()
            self.main_win.statusBar.showMessage(f"No BOS! No contours or more than 2 contours detected, repeat segmentation")
            return(mask)

    def stopThread(self):
        self.ThreadActive = False
        self.pipeline.stop()
        self.quit()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = mainWindow()
    main_win.show()
    sys.exit(app.exec_())


import tensorflow as tf
import cv2
import numpy as np
import random
import colorsys
from PIL import Image
import time
import pyrealsense2 as rs

interpreter = tf.lite.Interpreter(model_path="./model/yolov4-tiny-fp16.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

pipeline = rs.pipeline()
config = rs.config()
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 5)
spatial.set_option(rs.option.filter_smooth_alpha, 1)
spatial.set_option(rs.option.filter_smooth_delta, 50)

config.enable_stream(rs.stream.color, 848,480, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 848,480, rs.format.z16, 15)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
print(depth_scale)
time.sleep(1)
def nothing(x):
	pass
cv2.namedWindow('stream',cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('ROI',cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('accuracy','stream',50,100,nothing)


def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([416,416])):
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)

# def convert_depth_to_phys_coord_using_realsense(intrin,x, y, depth):  
#     result = rs.rs2_deproject_pixel_to_point(intrin, [x, y], depth)  
#     #result[0]: right (x), result[1]: down (y), result[2]: forward (z) from camera POV
#     return result[0], result[1], result[2]

names = {0: 'RF', 1: 'LF', 2: 'RA', 3: 'LA', 4: 'RT', 5: 'LT'}

def draw_bbox(thresh_image, white2, bboxes, classes=names, show_label=True):
    num_classes = len(classes)
    image_h, image_w = thresh_image.shape

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    white1 = np.zeros([480,848,3],dtype=np.uint8)
    white1.fill(255)
    
    for i in range(num_boxes[0]):
        if (int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes): continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)
        class_ind = int(out_classes[0][i])
        if (classes[class_ind] == 'RF' or classes[class_ind] == 'LF'):
            c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
            white1[c1[1]:c2[1],c1[0]:c2[0],2] = thresh_image[c1[1]:c2[1],c1[0]:c2[0]]
            # dilate = cv2.dilate(white1[:,:,2],None, iterations=4)
            # dilate = np.expand_dims(dilate,2)
            kernel = np.ones((3,3),np.uint8)
            open = cv2.morphologyEx(white1[:,:,2], cv2.MORPH_OPEN, kernel)
            contours, hierarchy = cv2.findContours(image=(255-open), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
            # areas = [cv2.contourArea(c) for c in contours]
            points = []
            if len(contours) == 2:
                for contour in contours:
                    # x,y,w,h = cv2.boundingRect(contour)
                    # cv2.rectangle(white2,(x,y),(x+w,y+h),(0,0,0),2)
                    # rect = cv2.minAreaRect(contour)
                    # box = cv2.boxPoints(rect)
                    # box = np.int0(box)
                    (x,y),radius = cv2.minEnclosingCircle(contour)
                    center = (int(x),int(y))
                    radius = int(radius)
                    cv2.circle(white2,center,radius,(0,0,0),2)
                    points.append((center,radius))
                # points.append((x+(w//2),y+(h//2)))
            if(len(points) == 2):
                cv2.line(white2, points[0][0], points[1][0], (240, 113, 57), 5)
                # for c,r in points:
                    
            cv2.drawContours(image=white2, contours=contours, contourIdx=-1, color=(210, 23, 165), thickness=2, lineType=cv2.LINE_AA)
            
            # cv2.drawContours(white2,points,0,(0,0,0),2)
        
    return np.hstack((white1,white2))


while True:
    frame = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    aligned_frame = align.process(frame)
    depth_frame = aligned_frame.get_depth_frame()
    color_frame = aligned_frame.get_color_frame()
    color_im = np.asanyarray(color_frame.get_data())
    # prof = depth_frame.get_profile()
    # video_prof = prof.as_video_stream_profile()
    # intrinsics = video_prof.get_intrinsics()
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)
    # color_image = cv2.flip(color_image, 1)
    blur = cv2.GaussianBlur(color_image,(5,5),0)
    gray_image = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    gray_image = np.expand_dims(gray_image,2)
    lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
    ret3,thresh = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # thresh = np.expand_dims(thresh, axis=2)
    # dilate = cv2.dilate(thresh,None, iterations=3)
    # dilate = np.expand_dims(dilate,2)
    # contours, hierarchy = cv2.findContours(image=dilate, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # img = np.zeros([480,848,3],dtype=np.uint8)
    # img.fill(255)
    # cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(25, 25, 112), thickness=2, lineType=cv2.LINE_AA)
    image_data = cv2.resize(color_image, (416,416))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    score = int(cv2.getTrackbarPos('accuracy','stream'))/100
    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=score, input_shape=tf.constant([416, 416]))

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=10,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=score
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    # white1 = np.zeros([480,848,3],dtype=np.uint8)
    white2 = np.zeros([480,848,3],dtype=np.uint8)
    white2.fill(255)
    
    white_im = draw_bbox(thresh, white2, pred_bbox)
    cv2.imshow('ROI', white_im)
    cv2.imshow('stream',color_im)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break
pipeline.stop()
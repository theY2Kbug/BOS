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

cv2.namedWindow('Controls',cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('accuracy','Controls',1,100,nothing)
cv2.createTrackbar("alpha", "Controls", 1, 65535, nothing)
# cv2.namedWindow('ROI', cv2.WINDOW_AUTOSIZE)


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

names = {0: 'RF', 1: 'LF', 2: 'RA', 3: 'LA', 4: 'RT', 5: 'LT'}

def draw_bbox(thresh_image, white, bboxes, classes=names, show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = thresh_image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    
    pred_class = []
    pred_score = []
    for i in range(num_boxes[0]):
        if (int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes): continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        # print(class_ind)
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
        white[c1[1]:c2[1],c1[0]:c2[0]] = thresh_image[c1[1]:c2[1],c1[0]:c2[0]]
    return white,pred_class,pred_score


while True:
    frame = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    aligned_frame = align.process(frame)
    depth_frame = aligned_frame.get_depth_frame()
    color_frame = aligned_frame.get_color_frame()
    color_im = np.asanyarray(color_frame.get_data())
    # depth_image = np.asanyarray(depth_frame.get_data())
    filtered_depth = spatial.process(depth_frame)
    depth_image = np.asanyarray(filtered_depth.get_data())
    # al = cv2.getTrackbarPos("alpha", "Controls")
    # final = cv2.convertScaleAbs(depth_image, alpha=(al/65535.0))
    
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # final = cv2.filter2D(final, -1, kernel)
    # ret3,th3 = cv2.threshold(final,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    color_image = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)
    # blur = cv2.GaussianBlur(color_image,(5,5),0)
    gray_image = cv2.cvtColor(color_im, cv2.COLOR_BGR2GRAY)
    # gray_image = cv2.equalizeHist(gray_image)
    twoDimage = color_im.reshape((-1,3))
    twoDimage = np.float32(twoDimage)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 2
    attempts=10
    ret,label,center=cv2.kmeans(twoDimage,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((color_im.shape))
    image_data = cv2.resize(color_image, (416,416))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    white = np.zeros([480,848,3],dtype=np.uint8)
    white.fill(255)
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    
    score = int(cv2.getTrackbarPos('accuracy','Controls'))/100
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
    white_im, classes,scores = draw_bbox(result_image, white, pred_bbox)
    
    img = np.zeros([480,848,3],dtype=np.uint8)
    img.fill(255)
    # white_im = 255 - white_im
    # print(classes)
    # print(scores)
    cv2.imshow('ROI', white_im)
    cv2.imshow('stream',color_im)
    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break
pipeline.stop()
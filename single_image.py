import tensorflow as tf
import cv2
import numpy as np
import random
import colorsys
from PIL import Image
import time

interpreter = tf.lite.Interpreter(model_path="./model/yolov4-tiny-832-bw-fp16.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
time.sleep(1)
cv2.namedWindow('Detect', cv2.WINDOW_AUTOSIZE)
# cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)

def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([832,832])):
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

def draw_bbox(image, bboxes, classes=names, show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
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
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        # pred_class.append(classes[class_ind])
        # pred_score.append(score)
        pred_class.append(classes[class_ind])
        pred_score.append(score)
        cv2.rectangle(image, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), bbox_color, bbox_thick)
        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, (int(c1[0]), int(c1[1])), (int(c3[0]), int(c3[1])), bbox_color, -1) #filled

            cv2.putText(image, bbox_mess, (int(c1[0]), int(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image,pred_class,pred_score



color_im = cv2.imread('./images/100-41.jpg')
color_image = cv2.cvtColor(color_im, cv2.COLOR_BGR2GRAY)
print(color_image.shape)
image_data = cv2.resize(color_image, (832,832))
image_data = np.expand_dims(image_data, axis=2)
image_data = image_data / 255.
image_data = image_data[np.newaxis, ...].astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], image_data)
interpreter.invoke()
pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([832, 832]))

boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
    scores=tf.reshape(
        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
    max_output_size_per_class=10,
    max_total_size=50,
    iou_threshold=0.45,
    score_threshold=0.15
)
pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
image,classes,scores = draw_bbox(color_im, pred_bbox)
print(classes)
print(scores)
cv2.imshow('Detect', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

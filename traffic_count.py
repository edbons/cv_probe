import cv2 as cv
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from sort import *


yolo_cfg_path = 'yolo'
conf_threshold = 0.7
NMS_threshold = 0.4

line_upper = []
line_lower = []

class_colors = {'car': (0, 255, 0), 'bus': (0, 0, 255), 'truck':(255, 0, 0)}
CLASSES = ['car', 'bus', 'truck']


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 255), 2, cv.LINE_AA)


def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def sort_predict(frame, detections, memory, counter, color, label, tracker):
    dets = np.asarray(detections)
    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    previous = memory.copy()
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

        if len(boxes) > 0:
            i = int(0)
            for box in boxes:

                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                cv.rectangle(frame, (x, y), (w, h), color, 2)

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                    p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                    cv.line(frame, p0, p1, color, 3)
                    if intersect(p0, p1, line_upper[0], line_upper[1]) | intersect(p0, p1, line_lower[0], line_lower[1]):                        
                        counter.add(indexIDs[i])
                        # print("id", indexIDs[i], len(counter))

                text = f"{label}"
                cv.putText(frame, text, (x, y - 5),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                i += 1
    
    return frame, memory, counter

def evaluate(path, save_video=False, show_gui=False):
    conf_threshold = 0.7
    NMS_threshold = 0.4

    class_colors = {'car': (0, 255, 0), 'bus': (0, 0, 255), 'truck':(255, 0, 0)}
    classes = ['car', 'bus', 'truck']

    with open(yolo_cfg_path + '/classes.txt', 'r') as f:
        class_names = [cname.strip() for cname in f.readlines()]

    class2idx = {name: class_names.index(name) for name in classes}
    idx2class = {v: k for k, v in class2idx.items()}

    net = cv.dnn.readNet(yolo_cfg_path + '/yolov4-tiny.weights',
                        yolo_cfg_path + '/yolov4-tiny.cfg')

    model = cv.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    cap = cv.VideoCapture(path)

    frame_w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    frame_h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    fourcc = cv.VideoWriter_fourcc(*'X264')
    # fourcc = cv.VideoWriter_fourcc(*'avc1')

    dim = (int(frame_w/4), int(frame_h/4))

    global line_upper
    global line_lower
    
    line_upper = [(0, int(dim[1] * (1/3))), (dim[0], int(dim[1] * (1/3))) ]
    line_lower = [(0, int(dim[1] * (2/3))), (dim[0], int(dim[1] * (2/3))) ]

    out = cv.VideoWriter('yolo_sort_output.mp4', fourcc, 15, dim, isColor=True)

    frame_counter = 0
    mot_tracker = Sort(max_age=2)
    memory = {}
    counter = set()
    label_counter = defaultdict(set)

    while True:
        ret, frame = cap.read()

        frame_counter += 1
        if ret == False:
            break

        frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
        classes, scores, yolo_boxes = model.detect(frame, conf_threshold, NMS_threshold)

        label_idxs = defaultdict(list)

        for i, value in enumerate(classes):
            if value in idx2class.keys():
                label_idxs[idx2class[value]].append(i)

        dets = []
        
        for label, idxs in label_idxs.items():
            if len(idxs) > 0:
                idxs = np.array(idxs)

                for i in idxs.flatten():            
                    (x, y) = (yolo_boxes[i][0], yolo_boxes[i][1])
                    (w, h) = (yolo_boxes[i][2], yolo_boxes[i][3])
                    dets.append([x, y, x+w, y+h, scores[i]])

                frame, memory, counter = sort_predict(frame, dets, memory, counter=label_counter[label], color=class_colors[label], label=label, tracker=mot_tracker)
                label_counter[label].update(counter)             

        cv.line(frame, line_upper[0], line_upper[1], (0, 255, 255), 2)
        cv.line(frame, line_lower[0], line_lower[1], (0, 255, 255), 2)

        draw_str(frame, (30, 40), f'car: {len(label_counter["car"])}')
        draw_str(frame, (30, 80), f'bus: {len(label_counter["bus"])}')
        draw_str(frame, (30, 120), f'truck: {len(label_counter["truck"])}')

        if show_gui:
            cv.imshow('frame', frame)
            key = cv.waitKey(30)
            if key == ord('q'):
                break

        if save_video:
            out.write(frame)
        
    out.release()

    cap.release()
    cv.destroyAllWindows()



def main():
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, help="path to video file")
    parser.add_argument('--save_video', action='store_true', help="save video with counts")    
    args = parser.parse_args()
    if args.save_video:
        evaluate(args.path, save_video=True)
        print("Predicts was saved to file yolo_sort_output.mp4")
    else:
        evaluate(args.path)


if __name__ == '__main__':
    main()

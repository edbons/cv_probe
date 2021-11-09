import cv2 as cv
import numpy as np
import pickle
import os
from argparse import ArgumentParser

with open('index2label.dict', 'rb') as f:
    index2label = pickle.load(f)

centroids = np.load('centroids.npy')

with open('basemodel.dict', 'rb') as f:
    base_model_edges = pickle.load(f)


def video_analyze(path: str, saved_path: str='./images', save_img: bool=False):
    video_name = os.path.basename(path).split('.')[0]

    feature_params = dict(maxCorners=500,
                      qualityLevel=0.1,
                      minDistance=2,
                      blockSize=7)

    lk_params = dict(winSize=(5, 5),
                    maxLevel=0,
                    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    cap = cv.VideoCapture(path)
    frame_size = (cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    points = []
    velocities = []
    
    ret, frame = cap.read()

    old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    points.append(p0.shape[0])
    
    while True:
        ret, frame = cap.read()

        if not ret:
            if save_img:
                for j, new in enumerate(good_new[good_points_idx]):
                    a, b = new.ravel()      
                    last_frame = cv.circle(old_gray, (int(a), int(b)), 5, (0, 0, 255), -1)

                output = f"{saved_path}/{video_name}.jpg"
                cv.imwrite(output, last_frame)
            break
        else:        
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            p1, st, err = cv.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, p0, None, **lk_params)
     
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                v = np.sqrt(np.sum( ((good_new - good_old) / frame_size) ** 2 , axis=1 ))  # V**2 = Vx**2 + Vy**2. Масштабируем делением на frame_size
                 
                good_points_idx = v > np.abs(np.mean(v) - np.std(v))  # фильтруем плохие объекты (например, неподвижные)

                good_points_velocity = v[good_points_idx]   
    
                if good_points_velocity.size > 0:
                    good_points_count = good_new[good_points_idx].shape[0]
                    points.append(good_points_count)
                    velocities.append(np.mean(good_points_velocity))
    
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
    
    cap.release()
    point_count = int(np.mean(points))
    velocity = np.mean(velocities) 

    return point_count, velocity


def predict(data: float, centroids: np.ndarray=centroids) -> str:
    diff = centroids - data  
    dist = np.sqrt(np.sum(diff ** 2, axis=-1)) 
    
    return index2label[np.argmin(dist)]


def predict_base(data: float) -> str:
    if base_model_edges['heavy'][0] < data < base_model_edges['heavy'][1]:
        pred = 'heavy'
    elif base_model_edges['medium'][0] < data < base_model_edges['medium'][1]:
        pred = 'medium'
    elif base_model_edges['light'][0] < data:
        pred = 'light'
    else:
        pred = None

    return pred 


def evaluate(path: str, saved_path: str='./images', save_img: bool=False, estimator='base') -> str:
    """
        Example: python -m traffic_intensity --path c:/datasets/car_traffic/video/cctv052x2004080516x01639.avi --save_img_path ./images --save_img --estimator kmeans
    """

    _, velocity = video_analyze(path, saved_path, save_img)
    if estimator == 'base':
        return predict_base(velocity)
    elif estimator == 'kmeans':
        return predict(np.expand_dims(np.array(velocity), axis=0), centroids).item()
    else:
        return None


def stream_predict_save(path: str, estimator='base'):
    feature_params = dict(maxCorners=1000,
                        qualityLevel=0.3,
                        minDistance=2,
                        blockSize=7)

    lk_params = dict(winSize=(5, 5),
                    maxLevel=0,
                    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    cap = cv.VideoCapture(path)

    new_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) / 3
    new_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) / 3

    fourcc = cv.VideoWriter_fourcc(*'X264')
    out = cv.VideoWriter('output.mp4', fourcc, 15, (int(new_w), int(new_h)), isColor=True)

    velocities = []
    text = ""
    ret, frame = cap.read()
    frame = cv.resize(frame, (int(new_w), int(new_h)), fx=0, fy=0, interpolation = cv.INTER_CUBIC)
    out.write(frame)

    old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    while cap.isOpened():
        ret, frame = cap.read()    

        if not ret:        
            break
        else:
            frame = cv.resize(frame, (int(new_w), int(new_h)), fx=0, fy=0, interpolation = cv.INTER_CUBIC)        
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            p1, st, err = cv.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, p0, None, **lk_params)
        
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                v = np.sqrt(np.sum( ((good_new - good_old) / (new_w, new_h)) ** 2 , axis=1 ))  # V**2 = Vx**2 + Vy**2. Масштабируем делением на frame_size
                    
                good_points_idx = v > np.abs(np.mean(v) - np.std(v))  # фильтруем плохие объекты (например, неподвижные)

                good_points_velocity = v[good_points_idx]   

                if good_points_velocity.size > 0:
                    velocities.append(np.mean(good_points_velocity))
                                    
                frame_number = cap.get(cv.CAP_PROP_POS_FRAMES) - 1

                for _, new in enumerate(good_new[good_points_idx]):
                        a, b = new.ravel()      
                        frame = cv.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
                
                if frame_number % 30 == 0:
                    p0 = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                    if estimator=='base':
                        text = predict_base(np.mean(velocities))
                    elif estimator=='kmeans':
                        text = predict(np.mean(velocities))
                    velocities = []
                
                else:                
                    p0 = good_new.reshape(-1, 1, 2)
                
                font = cv.FONT_HERSHEY_SIMPLEX
                frame = cv.putText(frame, text, (50, 50), font, 2, (0, 0, 255), 2, cv.LINE_AA)

            out.write(frame)
            old_gray = frame_gray.copy()
                
    cap.release()
    out.release()
    cv.destroyAllWindows()

def main():
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, help="path to video file")
    parser.add_argument('--save_img_path', type=str, default='./images', help="path for saving images")
    parser.add_argument('--save_img', action='store_true', help="save image to the save_img_path")
    parser.add_argument('--estimator', type=str, default='base', help="estimators: 'base', 'kmeans'")
    parser.add_argument('--stream', action='store_true', help="save video with predicts")    
    args = parser.parse_args()
    if args.stream:
        stream_predict_save(args.path, args.estimator)
        print("Predicts was saved to file output.mp4")
    else:
        print(evaluate(args.path, args.save_img_path, args.save_img, args.estimator))


if __name__ == '__main__':
    main()

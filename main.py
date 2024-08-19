import cv2
import numpy as np 
from util import get_parking_spots_bboxes , empty_or_not


def calc_diff(img1,img2):
    return np.abs(np.mean(img1) - np.mean(img2))


video_path = 'parking_1920_1080_loop.mp4'
mask = 'mask_1920_1080.png'


mask = cv2.imread(mask,0)
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

parking_spots = get_parking_spots_bboxes(connected_components)

spots_status = [None for j in parking_spots]

diffs = [None for j in parking_spots]

ret = True
step = 30
frame_n = 0

prev_frame = None
while ret :
    
    ret, frame = cap.read()

    if frame_n % step == 0 and prev_frame is not None :
        for spot_index,spot in enumerate (parking_spots):

            x1,y1,w,h  = spot 

            spot_crop = frame[y1:y1 +h , x1:x1 +w , :]

            diffs[spot_index] = calc_diff(spot_crop,prev_frame[y1:y1 +h , x1:x1 +w , :])            
                
    if frame_n % step == 0:
        if prev_frame is None:
             arr = range(len(parking_spots))
        else :
            arr = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_index in (arr):
            spot = parking_spots[spot_index]
            x1,y1,w,h  = spot 

            spot_crop = frame[y1:y1 +h , x1:x1 +w , :]

            spot_status = empty_or_not(spot_crop)
            
            spots_status[spot_index] = spot_status

    if frame_n % step == 0 :
        prev_frame = frame.copy()

    for spot_index,spot in enumerate (parking_spots):
        spot_status = spots_status[spot_index]
        x1,y1,w,h  = parking_spots[spot_index]
        if spot_status:

            frame = cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),2)

        else :
            frame = cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,0,255),2)

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
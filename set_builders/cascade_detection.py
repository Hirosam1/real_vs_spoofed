import os
import argparse
import numpy as np
import cv2
import numpy
#Parsing parametrts for __main__
def create_arg_parser():
    ap = argparse.ArgumentParser()

    ap.add_argument("-c","--cascade",
                    dest="c",
                    help="Path to the xml file, containing the cascaded to be used, can be more than one",
                    nargs="+",
                    required=True)

    args = ap.parse_args()
    return args

def detect(img,cascades,scaleFactor=1.1, minNeighbors=5,minSize=(80,80)):
    detections = []
    #tranforms to gray and equalize it
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    #for each cascade get the detections
    if isinstance(cascades,list):
        for cascade in cascades:
            detections.append(cascade.detectMultiScale(gray_eq, scaleFactor = scaleFactor, minNeighbors = minNeighbors, minSize = minSize))
    else:
        detections.append(cascades.detectMultiScale(gray_eq, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize))
    return detections

def main():
    args = create_arg_parser()
    cascades = []
    mini_img = None
    for c in args.c:
        #get files that ends with the .xml extension
        if c[-4:] == ".xml":
            cascades.append(cv2.CascadeClassifier(c))
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
       # mini_img = img
        detections = detect(img,cascades)
        count = 0
        #for each group of detections found by each cascade
        for detects in detections:
            #for each detection found by the cascade
            for (x, y, w, h) in detects:
                if count == 0:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)
                mini_img = img[y:y + h, x:x + w]
                cv2.rectangle(img, (x-2, y-2), (x + w+2, y + h+2), color, 2)
                cv2.putText(img,f">Img size: w{w} h{h}<",(x,y-2),cv2.FONT_HERSHEY_DUPLEX,1,color,1)
            count += 1
        cv2.imshow("img",img)
        if mini_img is not None:
            mini_img = cv2.resize(mini_img, (200, 200))
            cv2.imshow("only_face",mini_img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break;

    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
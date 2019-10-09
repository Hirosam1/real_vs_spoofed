from cascade_detection import detect
import os
import argparse
import cv2
import numpy

def find_max_w(place_holder):
    index = 0
    count = 0
    max = 0
    for a in place_holder:
        (x,y,w,h) = a
        if max < w:
            max = w
            index = count
        count += 1
    return index

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", dest="i",type=str, required=True,
	help="path the folder containing the images")
ap.add_argument("-o", "--output", dest="o",type=str, required=True,
	help="path to output directory of cropped faces")
ap.add_argument("-c", "--cascate", dest="c",type=str, required=True,
	help="Path to the xml file, containing the cascaded to be used")
ap.add_argument("-s", "--scale-factor", dest="scale",type=float, default=1.1,
	help="scale factor of the cascator")
ap.add_argument("-n", "--min-neighbors", dest="neighbors",type=float, default=5,
	help="min neighbors of the cascator")
ap.add_argument("-m", "--min-size", dest="m",type=int, default=[80,80], nargs=2,
	help="minimum size (X,X) of an image, anything smaller will be ignored ")
ap.add_argument("-r","--resize-img-to",dest="r",type=int, default=[300,300], nargs=2,
                help="The size that the images will be resized")
args = ap.parse_args()

photos = os.listdir(args.i)
cascade = cv2.CascadeClassifier(args.c)
for photo in photos:
    img = cv2.imread(os.path.join(args.i, photo))
    place_holder = detect(img,cascade,scaleFactor=args.scale,minNeighbors=args.neighbors,minSize=tuple(args.m))[0]
    if place_holder is not ():
        i = find_max_w(place_holder)
        (x,y,w,h) = place_holder[i]
        face = img[y:y+h,x:x+h]
        face = cv2.resize(face,tuple(args.r))
        p = os.path.join(args.o, f"{photo}")
        cv2.imwrite(p, face)
        print(f"Photo {photo} saved sucessefully!")
    else:
        print(f"No detection for photo {photo}!!")

cv2.destroyAllWindows()


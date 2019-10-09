import cv2
import os
import argparse
from cascade_detection import detect

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input",dest="i", required=True)
    parser.add_argument("-o","--output",dest="o",required=True)
    parser.add_argument("-c","--cascade", dest="c",required=True)
    parser.add_argument("-s","--skip-frames",dest="s",default=5, type=int)
    parser.add_argument("-r","--resize-image", dest = "r", default=[300,300],nargs=2)
    parser.add_argument("-l","--limit-images-person-person", dest="l",default=6000, type = int)

    args = parser.parse_args()
    return args


def get_imgs(vc,cascade,output, name,skip=60,max_images=1500, count= 0):
    (grabbed, frame) = vc.read()
    read = 0
    write =0
    while grabbed:
        read += 1
        if read % int(skip) == 0:
            face = get_face(cascade,frame)
            if face is not None:
                write += 1
                count += 1
                this_output = os.path.join(output,f"{name}{write}.png")
                cv2.imwrite(this_output,face)
                if count >= max_images:
                    return write
                    print("")
        print(f"\r\tGot {write} faces \tFrames: {read}", end="")
        (grabbed, frame) = vc.read()



    print("")
    return write


def get_face(cascade,img):
    place_holder = detect(img, cascade,scaleFactor=1.1, minNeighbors=5, minSize=(80,80))[0]
    if place_holder is not ():
        i = find_max_w(place_holder)
        (x,y,w,h) = place_holder[i]
        face = img[y:y+h,x:x+h]
        face = cv2.resize(face,tuple(args.r))
        return face

if __name__ == "__main__":
    args = parse_args()
    people = os.listdir(args.i)
    cascade = cv2.CascadeClassifier(args.c)
    for person in people:
        person_imgs = 0
        out_path = os.path.join(args.o,person)
        os.mkdir(out_path)
        person_path = os.path.join(args.i,person)
        print(f"Getting {person} face photos")
        videos = os.listdir(person_path)
        for video in videos:
            if person_imgs >= args.l:
                continue
            if video[-4:] == "face":
                continue

            vc = cv2.VideoCapture(os.path.join(person_path,video))
            if not vc.isOpened():
                print(f"Ignoring this file: {video}")
                continue
            writes = get_imgs(vc,cascade,out_path,video,args.s,args.l, person_imgs)
            person_imgs += writes


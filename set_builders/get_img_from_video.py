import cv2
import os
import argparse


def create_arg_parser():
    ag = argparse.ArgumentParser()

    ag.add_argument("--videos-input",
                    dest="vi",
                    help="Video folders containig folders of videos",
                    required=True)
    ag.add_argument("--photos-output",
                    dest="po",
                    help="folder for the output of images",
                    required=True)
    ag.add_argument("--skip", dest="skip",help="Numbers of frames to skip between each photo", default=90)

    args = ag.parse_args()
    return args
def get_frames (vc,output,skip):
    (grabbed,frame) = vc.read()
    read = 0
    write = 0
    while grabbed:
        read += 1
        if read % int(skip) == 0:
            write +=1
            this_output = output + f"{write}.png"
            cv2.imwrite(this_output,frame)

        (grabbed, frame) = vc.read()
    return write

def main():
    args = create_arg_parser()
    people = os.listdir(args.vi)

    for person in people:
        person_path = os.path.join(args.vi, person)
        videos = os.listdir(person_path)
        print(f"Getting {person} photos:")
        count = 0
        for video in videos:
            if video[-4:] == "face":
                continue
            vc = cv2.VideoCapture(os.path.join(person_path, video))
            if not vc.isOpened():
                print(f"Ignoring this file: {video}")
                continue

            writes = get_frames(vc, os.path.join(args.po, f"{person}_{count}_"), args.skip)
            print(f"\t->Got {writes} photos")
            count += 1
if __name__ == "__main__":
    main()

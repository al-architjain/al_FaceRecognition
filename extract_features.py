#python extract_features.py --dataset recoginisedImages --

#Importing Modules
import os
import cv2
import pickle
import argparse
from imutils import paths
import face_recognition


#Arguments importing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
# ap.add_argument("-d", "--detection-method", type=str, default="hog", help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


class Encodings():

    def __init__(self):
        self.imagePath = list(paths.list_images(args["dataset"]))
        self.knownEncodings = []
        self.knownNames = []

    def Xtract(self):
        print("Xtracting features :)")
        for (i, imagePath) in enumerate(self.imagePath):

            print("[INFO] processing image {}/{}".format(i + 1, len(self.imagePath)))
            name = imagePath.split(os.path.sep)[-2]

            # Read Image
            image_bgr = cv2.imread(imagePath)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # coz OpenCV stores in BGR :(

            # Detection, Landmark detection, and encoding
            locations = face_recognition.face_locations(image_rgb, model='hog')  # or (model=args["detection_method"])
            landmarks = face_recognition.face_landmarks(image_rgb)
            encodings = face_recognition.face_encodings(image_rgb, locations)

            # Store Encodings (locally)
            for encoding in encodings:
                self.knownEncodings.append(encoding)
                self.knownNames.append(name)

    def store(self):
        # Store encodings (globally)
        print("[INFO] serializing encodings...")
        data = {"encodings": self.knownEncodings, "names": self.knownNames}
        f = open(args["encodings"], "wb")
        f.write(pickle.dumps(data))
        f.close()


def Main():
    eobj = Encodings()
    eobj.Xtract()
    eobj.store()


if __name__=='__main__':
    Main()

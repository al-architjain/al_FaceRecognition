'''
    How to Run:
    python extractFeatures.py --dataset dataset --encodingsfile encoding.pickle
'''

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
ap.add_argument("-e", "--encodingsfile", required=True, help="path to serialized db of facial encodings")
# ap.add_argument("-d", "--detection-method", type=str, default="hog", help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


class featureExtraction():

    def __init__(self):
        self.imagePath = list(paths.list_images(args["dataset"]))
        self.knownEncodings = []
        self.knownNames = []

    def Xtract(self):
        print("Xtracting features :)")
        lastname = "Unknown"
        for (i, imagePathIterator) in enumerate(self.imagePath):
            name = imagePathIterator.split(os.path.sep)[-2]
            if name != lastname:
                lastname = name
                print("[Info] Encodings for " + name)
            # Read Image
            print("[INFO] processing image {}/{}".format(i + 1, len(self.imagePath)))
            image_bgr = cv2.imread(imagePathIterator)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
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
        f = open(args["encodingsfile"], "wb")
        f.write(pickle.dumps(data))
        f.close()
        print("[INFO] Encoding Complete!")


def Main():
    eobj = featureExtraction()
    eobj.Xtract()
    eobj.store()


if __name__=='__main__':
    Main()

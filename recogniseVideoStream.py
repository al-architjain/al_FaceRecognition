'''
    How to Run:
    python recognizeVideoStream.py --encodings encodings.pickle
'''

#Import Modules
import cv2
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time

#Importing Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
# ap.add_argument("-o", "--output", type=str, help="path to output video")
# ap.add_argument("-y", "--display", type=int, default=1, help="whether or not to display output frame to screen")
# ap.add_argument("-d", "--detection-method", type=str, default="hog", help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


#Video Stream Recognition Class
class recogniseVideoStream():

    def __init__(self):
        #Loading Data
        print("[INFO] Initializing variables and Loading encodings...")
        self.known_data = pickle.loads(open(args["encodings"], "rb").read())

        self.known_name_counts = {}
        for iname in self.known_data["names"]:
            self.known_name_counts[iname] = self.known_name_counts.get(iname, 0) + 1


    def recogniseImage(self,encodings):
        names = []
        for encoding in encodings:
            matches = face_recognition.compare_faces(self.known_data["encodings"], encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                for i in matchedIdxs:
                    iname = self.known_data["names"][i]
                    counts[iname] = counts.get(iname, 0) + 1

                maxname = max(counts, key=counts.get)
                maxnameperc = (counts[maxname] * 100) / self.known_name_counts[maxname]

                if maxnameperc >= 40:
                    name = maxname + str(maxnameperc) + "%"
                else:
                    name = "Unknown"

            # update the list of names
            names.append(name)
        return names


    def startStreaming(self):
        print("[INFO] Starting Video Stream...")
        streamObj = VideoStream(src=0).start()
        time.sleep(1.0)

        while True:
            frame = streamObj.read()

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb = imutils.resize(image_rgb, width=750)
            r = frame.shape[1] / float(image_rgb.shape[1])

            #Detecting parameters
            locations = face_recognition.face_locations(image_rgb, model='hog')  # (model=args["detection_method"])
            landmarks = face_recognition.face_landmarks(image_rgb)
            encodings = face_recognition.face_encodings(image_rgb, locations)

            names = self.recogniseImage(encodings)

            # loop over the recognized faces
            for ((top, right, bottom, left), name) in zip(locations, names):
                # rescale the face coordinates
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)

                cv2.rectangle(frame, (left, top), (right, bottom + 35), (0, 0, 255), 2)
                # y = top - 15 if top - 15 > 15 else top + 15
                cv2.rectangle(frame, (left, bottom), (right, bottom + 35), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 2, bottom + 30), font, 0.85, (255, 255, 255), 1)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        streamObj.stop()


def Main():
    vobj = recogniseVideoStream()
    vobj.startStreaming()


if __name__=='__main__':
    Main()

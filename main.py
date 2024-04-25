from ultralytics import YOLO
import cv2
import math
import ntcore

inst = ntcore.NetworkTableInstance.getDefault()
table = inst.getTable("personDetector")
xEntry = table.getEntry("personX")
personCountEntry = table.getEntry("personCount")
personListEntry = table.getEntry("personList")
# xSub = table.getDoubleTopic("x").subscribe(0)
# ySub = table.getDoubleTopic("y").subscribe(0)
inst.startClient4("personDetector")
inst.setServerTeam(4509)  # where TEAM=190, 294, etc, or use inst.setServer("hostname") or similar
# inst.setServer("localhost")
# inst.startDSClient()  # recommended if running on DS computer; this gets the robot IP from the DS


# start webcam
cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_EXPOSURE, 50)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")


# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    print("ntConn"+ str(inst.isConnected()))
    success, img = cap.read()
    results = model(img, stream=True, imgsz=320)
    personCount = 0

    # coordinates
    for r in results:
        boxes = r.boxes

        peopleXVals = []

        for box in boxes:
            # class name
            cls = int(box.cls[0])
          #  print("Class name -->", classNames[cls])

            if classNames[cls] != "person":
                break
            personCount += 1

            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
         #   print("Confidence --->", confidence)


            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            # cv2.putText(img, classNames[cls] + ", conf%:" + str(confidence), org, font, fontScale, color, thickness)
            cv2.putText(img, str(confidence)+"%", org, font, fontScale, color, thickness)
            centerX = float((((box.xyxy[0][0] + box.xyxy[0][2]) / 2) - 320 )/ 640)
            peopleXVals.append(centerX)
            xEntry.setDouble(centerX)

        personListEntry.setDoubleArray(peopleXVals)
        personCountEntry.setInteger(personCount)



    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2

# From Image
def ImgFile():
   # Read image from file
   img = cv2.imread('Human.jpg')

   # Load class names from coco.names file
   classNames = []
   classFile = r"g:/Final Projects/objective_detectiion/python-object-detection-opencv-main/files/coco.names"  # adjust your path
   with open(classFile, 'rt') as f:
      classNames = f.read().rstrip('\n').split('\n')

   # Load pre-trained model and configure it
   configPath = r'g:/Final Projects/objective_detectiion/python-object-detection-opencv-main/files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
   weightpath = r'g:/Final Projects/objective_detectiion/python-object-detection-opencv-main/files/frozen_inference_graph.pb'
   net = cv2.dnn_DetectionModel(weightpath, configPath)
   net.setInputSize(320, 230)
   net.setInputScale(1.0 / 127.5)
   net.setInputMean((127.5, 127.5, 127.5))
   net.setInputSwapRB(True)

   # Detect objects in the image
   classIds, confs, bbox = net.detect(img, confThreshold=0.5)
   print(classIds, bbox)

   # Draw bounding boxes and labels on the image
   for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
      cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
      cv2.putText(img, classNames[classId-1], (box[0] + 10, box[1] + 20), 
                  cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

   # Display the output image
   cv2.imshow('Output', img)
   cv2.waitKey(0)   # (0): is the camera of the laptop

# From Video or Camera 
def Camera():
   # Open video capture (0 for default camera)
   cam = cv2.VideoCapture(0)

   # Set the resolution of the camera
   cam.set(3, 740)
   cam.set(4, 580)

   # Load class names from coco.names file
   classNames = []
   classFile = r'D:/NTI Projects/project 2/python-object-detection-opencv-main/coco.names'
   with open(classFile, 'rt') as f:
      classNames = f.read().rstrip('\n').split('\n')

   # Load pre-trained model and configure it
   configPath = r'D:/NTI Projects/project 2/python-object-detection-opencv-main/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
   weightpath = r'D:/NTI Projects/project 2/python-object-detection-opencv-main/frozen_inference_graph.pb'
   net = cv2.dnn_DetectionModel(weightpath, configPath)
   net.setInputSize(320 , 230)
   net.setInputScale(1.0 / 127.5)
   net.setInputMean((127.5, 127.5, 127.5))
   net.setInputSwapRB(True)

   # Process frames from the camera
   while True:
      # Read a frame from the camera
      success, img = cam.read()
      if not success:
         print("Failed to read frame from camera")
         break

      # Detect objects in the frame
      classIds, confs, bbox = net.detect(img, confThreshold=0.5)
      print(classIds, bbox)

      # Draw bounding boxes and labels on the frame
      if len(classIds) != 0:
         for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId-1], (box[0] + 10, box[1] + 20), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

      # Display the output frame
      cv2.imshow('Output', img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
         break

   # Release the camera and close all OpenCV windows
   cam.release()
   cv2.destroyAllWindows()

# Call ImgFile() Function for Image Or Camera() Function for Video and Camera
# ImgFile()
Camera()

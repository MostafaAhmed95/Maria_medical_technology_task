# license removed for brevity
import rospy
from std_msgs.msg import String
import cv2 
import dlib

pub = rospy.Publisher('chatter', String, queue_size=10)
rospy.init_node('talker', anonymous=True)
rate = rospy.Rate(10) # 10hz

def draw_grid(img,left_corner,right_corner,span):
    
    for i in range (span,right_corner[1],span):
        if (left_corner[1]+i < right_corner[1]):
            cv2.line(img,(left_corner[0],left_corner[1]+i),(right_corner[0],left_corner[1]+i),(0,255,0),4)
        if (left_corner[0]+i < right_corner[0]):
            cv2.line(img,(left_corner[0]+i,left_corner[1]),(left_corner[0]+i,right_corner[1]),(0,255,0),4)
            #print("done")

def free(*x):
    pass

span = 10
x1,y1,widthy = 0,0,0
det = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facecas = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
cv2.namedWindow('face')
cv2.createTrackbar('R','face',10,50,free)
cap = cv2.VideoCapture(0)


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces_dlib = det(gray)
    faces_harr = facecas.detectMultiScale(gray)
    
    for face in faces_harr:
        x1, y1, w, h = face
        widthy = x1+w
    

    for face in faces_dlib:
        landmarks = predict(image=gray, box=face)
        x1 = landmarks.part(17).x
        x = landmarks.part(26).x
        y = landmarks.part(26).y

        cv2.rectangle(frame,(x1,y1),(x,y),(0,255,0),4)
        area = (x-x1)*(y-y1)
        pub.publish(str(area))
        #cv2.line(frame,(x1,y1),(x,y),(0,255,0),4)
        draw_grid(frame, (x1,y1), (x,y),span)

    cv2.imshow("face",frame)
    span = int(cv2.getTrackbarPos('R','face'))
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
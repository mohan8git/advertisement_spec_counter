import cv2
import face_recognition
from timeit  import default_timer as timer
start = timer()
frame = cv2.imread(r"C:\Users\dapz\Desktop\Face Recognition\Face Encoding\Images\image1.jpg")


def FindPoint(x1, x2, y1,  
              y2, cx, cy) : 
    if (cx > x1 and cx < x2 and 
        cy > y1 and cy < y2) : 
        return True
    else : 
        return False



o = []
i = 0
face_locations = face_recognition.face_locations(frame)
for top, right, bottom, left in face_locations:
    i = i+1        
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    y = bottom + int((top-bottom)*0.5)
    x = left + int((right - left)*0.5)
    if FindPoint(left, right, top,  
                 bottom, x, y) : 
        print("Yes") 
    else : 
        print("No") 

    cv2.circle(frame,(x,y,),5 , (0,255,255), -1)
key = cv2.waitKey(1)
if key == ord('q'):
    image.release()
    
    cv2.destroyAllWindows()
    

print(timer()-start)


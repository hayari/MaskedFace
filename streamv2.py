import numpy as np
import cv2
from keras.models import load_model
from mtcnn import MTCNN


cap = cv2.VideoCapture(0)

#load the model
model=load_model('my_model_128.h5',custom_objects=None,compile=True)

detector = MTCNN()

font = cv2.FONT_HERSHEY_SIMPLEX
green=(0,255,0)
red=(0,0,255)
  
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    #draw_facebox
    
    faces = detector.detect_faces(img)
    
    
    for onebox in faces:
        x, y, w, h=onebox['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        oneface = img[y:y+h, x:x+w]
        oneface = cv2.resize(oneface,(128,128))
        # predict class
        pclass=int(model.predict(oneface[np.newaxis,:,:,:])[0])
        #print(pclass)
        # print class
        if pclass==0:
                cv2.putText(frame,'Unmasked',(x,y), font, 0.7,red,2,cv2.LINE_AA)
        else:
                cv2.putText(frame,'masked',(x,y), font, 0.7,green,2,cv2.LINE_AA)
        
        
    
    
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

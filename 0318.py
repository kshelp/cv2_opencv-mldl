#0318.py
import cv2
import numpy as np

#1
model = cv2.ml_ANN_MLP.load('./data/0315_ANN_RPROP.train')
#model = cv2.ml_ANN_MLP.load('./data/0316_ANN_BACKPROP.train')

#2:
def makeSquareImage(img):
    height, width  = img.shape[:2]
    if width > height:
        y0 = (width-height)//2
        resImg = np.zeros(shape=(width, width), dtype=np.uint8)
        resImg[y0:y0+height, :] = img

    elif width < height:
        x0 = (height-width)//2
        resImg = np.zeros(shape=(height, height), dtype=np.uint8)
        resImg[:, x0:x0+width] = img
    else:
        resImg = img
    return resImg

#3
colors = {'black':(0,0,0),    'white': (255, 255, 255),
          'blue': (255,0,0) , 'red':   (0,0,255)}

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(src, (x, y), 10, colors['black'], -1) # draw
            cv2.imshow('image', src)
        elif flags & cv2.EVENT_FLAG_RBUTTON:
            cv2.circle(src, (x, y), 10, colors['white'], -1) # erase
            cv2.imshow('image', src)
 
src  = np.full((512, 512, 3), colors['white'], dtype=np.uint8)
cv2.imshow('image', src)
cv2.setMouseCallback('image', onMouse)
   
font = cv2.FONT_HERSHEY_SIMPLEX  
x_img = np.zeros(shape=(28, 28), dtype=np.uint8)

#4
while True:
#4-1
    key = cv2.waitKey(25)
    if   key == 27: # esc
        break;
    elif key == 32: # space
        src[:,:] = colors['white'] # clear image
        cv2.imshow('image',src)
#4-2        
    elif key == 13: # return  
        print("----classify....")
        dst = src.copy()
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        ret, th_img = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        
#4-3:
        for i, cnt in enumerate(contours):
            x, y, width, height = cv2.boundingRect(cnt)
            area = width*height
            if area < 1000: # too small
                continue
            
            cv2.rectangle(dst, (x, y), (x+width, y+height), colors['red'], 2)
            
            x_img[:,:] = 0       # black background 

            img = th_img[y:y+height, x:x+width]
            img = makeSquareImage(img)
            
            img = cv2.resize(img, dsize=(20, 20),interpolation=cv2.INTER_AREA)            
            x_img[4:24, 4:24] = img
            x_img = cv2.dilate(x_img, None, 2)
            x_img = cv2.erode(x_img,  None, 4)
 
#4-4: predict
            x_test = np.float32(x_img/255).reshape(1, -1) # (1, 784)
            ret, pred = model.predict(x_test) 
            digit = np.argmax(pred, axis = 1)[0]
            print('digit=', digit)           

            cv2.putText(dst, str(digit), (x, y), font, 2, colors['blue'], 3)
        
        cv2.imshow('image', dst)
cv2.destroyAllWindows()

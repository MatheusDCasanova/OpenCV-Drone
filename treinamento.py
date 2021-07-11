import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("D:\programacao\ger\Percepcao\grame0042.jpg")

cv.imshow("imagem", img)
cv.waitKey(0) 

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

min_red = np.array([0,50,200])
max_red = np.array([10,255,255])

red_mask = cv.inRange(hsv, min_red, max_red)

min_green = np.array([55, 50, 200])
max_green = np.array([65, 255, 255])

green_mask = cv.inRange(hsv, min_green, max_green)

red_img = cv.bitwise_and(img,hsv, mask= red_mask)
green_img = cv.bitwise_and(img, hsv, mask=green_mask)

kernel = np.ones((3,3),np.uint8)

red_img = cv.morphologyEx(red_img, cv.MORPH_CLOSE, kernel)
red_img = cv.morphologyEx(red_img, cv.MORPH_OPEN, kernel)
green_img = cv.morphologyEx(green_img, cv.MORPH_OPEN, kernel)

cv.imshow("red_img", red_img)
cv.waitKey(0) 
cv.imshow("green_img", green_img)
cv.waitKey(0) 

red_bordas = cv.Canny(red_img, 100, 200)
green_bordas = cv.Canny(green_img, 300, 300)
red_quinas = cv.goodFeaturesToTrack(red_bordas, 8, 0.05, 4)
red_quinas = np.int0(red_quinas)

'''lista_de_quinas = []

for ponto in red_quinas:
    x,y = ponto.ravel()
    print([x,y])
    lista_de_quinas.append([x,y])
    cv.circle(red_bordas, (x,y), 2, (255,255,255), -1)

print(len(red_contorno))'''

red_contorno, hierarquia = cv.findContours(red_bordas, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for contorno in red_contorno:
    area = cv.contourArea(contorno)
    print(area)
    if (area >= 25):
        aprox = cv.approxPolyDP(contorno, 0.009 * cv.arcLength(contorno, True), True)
        print(len(aprox))
        if(len(aprox) > 4): 
            cv.drawContours(green_bordas, [aprox], -1, (255, 255, 255), 2)


plt.imshow(red_bordas)
plt.show()

plt.imshow(green_bordas)
plt.show()


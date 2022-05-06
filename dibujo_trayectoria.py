
import cv2
import numpy as np
import os
import json
points = []
flag = False
def click(event, x, y, flags, param):
    global flag, points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        flag = True


def recortar_imagen(imagen):
    puntos = np.where((imagen[:,:,0]>0) * (imagen[:,:,1]>0)* (imagen[:,:,2])>0)
    if(puntos[0].shape[0]>0):
                max_y = max(puntos[0])
                max_x = max(puntos[1])
                min_y = min(puntos[0])
                min_x = min(puntos[1])
                img_guar = imagen[min_y:max_y,min_x:max_x,:]  
                return img_guar,min_x,min_y
    else:
        return -1,-1,-1


def dibujo_trayectoria(image,nombre):
    
    global flag, points, resolucion, pos_x_map, pos_y_map
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click)
    image_draw= image.copy()
    points1 = []
    while True:
        cv2.imshow("Image", image_draw)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            cv2.destroyAllWindows()
            break
        elif(key == ord("c")):
            flag = 1
        if flag:
            flag = False
            points1.pop(-1)
            image_draw= image.copy()
            [cv2.circle(image_draw, (punto[0], punto[1]), 3, [0, 0, 255], -1) for punto in points1[::2] ]
            [cv2.circle(image_draw, (punto[0], punto[1]), 3, [255, 0, 0], -1) for punto in points1[1::2] ]
            
        if len(points) > 0:
            points1.append((points[0][0],points[0][1]))
            points= []
        
            image_draw= image.copy()
            [cv2.circle(image_draw, (punto[0], punto[1]), 3, [0, 0, 255], -1) for punto in points1[::2] ]
            [cv2.circle(image_draw, (punto[0], punto[1]), 3, [255, 0, 0], -1) for punto in points1[1::2] ]
            
    file = open(nombre+str(1)+".txt", "w")
    [file.write(str((i[0]-pos_x_map)/resolucion)+"  "+ str((i[1]-pos_y_map)/resolucion)+"\n") for i in points1[::2]]

    file.close()
    
    
    file = open(nombre+str(2)+".txt", "w")

    [file.write(str((i[0]-pos_x_map)/resolucion)+"  "+ str((i[1]-pos_y_map)/resolucion)+"\n") for i in points1[1::2]]

    file.close()
    


mapa = cv2.imread("mapa/mapa_fabrica_interp.png")


with open('mapa/mapa_fabrica_interp.json') as file:
    data = json.load(file)
    
resolucion = data["datos_mapa"][0]["resolucion"]
pos_x_map = data["datos_mapa"][0]["pos_x_mapa"]
pos_y_map = data["datos_mapa"][0]["pos_y_mapa"]

cv2.circle(mapa, (pos_x_map,pos_y_map), 3, [0, 255, 255], -1)


dibujo_trayectoria(mapa,"mapa/path_robot")




















"""

mapa,delta_x,delta_y = recortar_imagen(mapa)
pos_x_map = (pos_x_map-delta_x)*3
pos_y_map = (pos_y_map-delta_y)*3
###cv2.circle(mapa, (pos_x_map, pos_y_map), 3, [0, 0, 255], -1)


resized = cv2.resize(mapa, (mapa.shape[0]*3,mapa.shape[1]*3), interpolation = cv2.INTER_AREA)


cv2.imshow("imagen_2",resized)
cv2.waitKey(0)


cv2.imwrite("mapa/mapa_fabrica_interp.png",resized)



data = {}
data['datos_mapa'] = []
data['datos_mapa'].append({
    'resolucion': resolucion,
    'pos_x_mapa': int(pos_x_map),
    'pos_y_mapa': int(pos_y_map)})

with open('mapa/mapa_fabrica_interp.json', 'w') as file:
    json.dump(data, file, indent=4)

"""




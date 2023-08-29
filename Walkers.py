import cv2
import numpy as np #importando numpy


# Crie nosso classificador de corpos
body_classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")#classificador de corpos


# Inicie a captura de vídeo para o arquivo de vídeo
cap = cv2.VideoCapture('walking.avi')

# Faça o loop assim que o vídeo for carregado com sucesso
while True:
    
    # Leia o primeiro quadro
    ret, frame = cap.read()

    # Converta cada quadro em escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Transformador de escala de cinza
    # Passe o quadro para nosso classificador de corpos
    bodies = body_classifier.detectMultiScale(gray,1.2,3) # detectador em multiplas escalas
    
    # Extraia as caixas delimitadoras para quaisquer corpos identificados
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(200,123,0),6) #retangulo criado x,y cordenada w,h altura e largura
        cv2.imshow("pedestres",frame)

    if cv2.waitKey(1) == 32: #32 é a barra de espaço
        break

cap.release()
cv2.destroyAllWindows()

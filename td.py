import cv2
import numpy as np

# Charger les classificateurs Haar pour les visages et les yeux
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye_tree_eyeglasses.xml')

# Charger la vidéo et les images des accessoires
video = cv2.VideoCapture(0)
santa_hat = cv2.imread('./images/chapeau.png', -1)  # Bonnet avec canal alpha
sunglasses = cv2.imread('./images/sunglasses.png', -1)  # Lunettes avec canal alpha

# Variables pour activer/désactiver les filtres
show_hat = True
show_glasses = True

# Gestionnaire d'événements pour détecter les clics de souris
def mouse_event(event, x, y, flags, param):
    global show_hat, show_glasses
    height, width = param['frame_height'], param['frame_width']
    
    # Vérifier si un clic a lieu sur les boutons
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 110 and height - 80 <= y <= height - 40:  # Bouton "Bonnet"
            show_hat = not show_hat
        elif 10 <= x <= 110 and height - 40 <= y <= height:  # Bouton "Lunettes"
            show_glasses = not show_glasses

# Créer la fenêtre pour afficher la vidéo
cv2.namedWindow('Esprit de Noël')

while video.isOpened():
    # Lire une frame de la vidéo
    ret, frame = video.read()
    if not ret:
        break

    # Convertir la frame en niveaux de gris pour la détection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Dimensions de la frame
    height, width, _ = frame.shape

    # Dessiner les boutons
    cv2.rectangle(frame, (10, height - 80), (110, height - 40), (0, 255, 0), -1)
    cv2.putText(frame, 'Bonnet', (15, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.rectangle(frame, (10, height - 40), (110, height), (0, 255, 255), -1)
    cv2.putText(frame, 'Lunettes', (15, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    for (x, y, w, h) in faces:
        # --------- BONNET ---------
        if show_hat:
            resized_santa_hat = cv2.resize(santa_hat, (w, int(w * santa_hat.shape[0] / santa_hat.shape[1])))
            hat_height, hat_width, _ = resized_santa_hat.shape

            hat_top_y = max(y - int(h / 3), 0)  # Positionner légèrement au-dessus des sourcils
            roi_hat = frame[hat_top_y:hat_top_y + hat_height, x:x + hat_width]

            for i in range(hat_height):
                for j in range(hat_width):
                    if resized_santa_hat[i, j, 3] != 0:  # Vérifier si le pixel n'est pas transparent
                        roi_hat[i, j] = resized_santa_hat[i, j, :3]

        # --------- LUNETTES ---------
        if show_glasses:
            resized_sunglasses = cv2.resize(sunglasses, (w, h // 3))
            sg_height, sg_width, _ = resized_sunglasses.shape
            roi = frame[y + h // 4:y + h // 4 + sg_height, x:x + sg_width]

            for i in range(sg_height):
                for j in range(sg_width):
                    if resized_sunglasses[i, j, 3] != 0:  # Vérifier si le pixel n'est pas transparent
                        roi[i, j] = resized_sunglasses[i, j, :3]

    # Définir les dimensions de la frame pour les événements
    cv2.setMouseCallback('Esprit de Noël', mouse_event, param={'frame_height': height, 'frame_width': width})
    
    # Afficher la vidéo avec les accessoires incrustés
    cv2.imshow('Esprit de Noël', frame)

    # Quitter si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
video.release()
cv2.destroyAllWindows()

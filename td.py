import cv2
import numpy as np

# Charger le classificateur en cascade pour la détection de visage
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

# Charger l'image de la barbe (assurez-vous que l'image est transparente PNG)
beard_img = cv2.imread('./images/beard.png', cv2.IMREAD_UNCHANGED)

while True:
    # Lire une image depuis la webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en niveaux de gris pour la détection de visage
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Pour chaque visage détecté
    for (x, y, w, h) in faces:
        # Définir les dimensions de la barbe (ajustez selon votre image de barbe)
        beard_width = w
        beard_height = int(beard_width / beard_img.shape[1] * beard_img.shape[0])

        # Redimensionner l'image de la barbe
        beard_resized = cv2.resize(beard_img, (beard_width, beard_height))

        # Obtenir les coordonnées pour superposer la barbe
        x_offset = x
        y_offset = y + int(h / 1.5)  # Positionner la barbe sous le menton

        # Extraire les canaux de l'image de la barbe (RGBA)
        beard_alpha = beard_resized[:, :, 3] / 255.0  # Canal alpha (transparence)

        # Superposer la barbe sur l'image du visage
        for c in range(0, 3):
            frame[y_offset:y_offset+beard_height, x_offset:x_offset+beard_width, c] = \
                (1. - beard_alpha) * frame[y_offset:y_offset+beard_height, x_offset:x_offset+beard_width, c] + \
                beard_alpha * beard_resized[:, :, c]

    # Afficher l'image avec la barbe ajoutée
    cv2.imshow('Face with Beard', frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import tkinter
import time

# Charger les images pour l'incrustation
bonnet = cv2.imread('./images/chapeau.png', cv2.IMREAD_UNCHANGED)  # Bonnet avec transparence
lunettes = cv2.imread('./images/sunglasses.png', cv2.IMREAD_UNCHANGED)  # Lunettes pixelisées
background_image = cv2.imread('decor.jpg')  # Fond personnalisé (à. préparer)

# Détecteur de visage
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

# Fonction pour appliquer un filtre sépia
def apply_sepia(frame):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    return cv2.transform(frame, sepia_filter)

# Fonction pour incruster une image avec transparence
def overlay_image(background, overlay, x, y):
    h, w, _ = overlay.shape
    for i in range(h):
        for j in range(w):
            if overlay[i, j, 3] > 0:  # Canal alpha
                background[y + i, x + j] = overlay[i, j, :3]
    return background

# Fonction pour incruster des flocons de neige
def add_snowflakes(frame, snowflakes):
    for snow in snowflakes:
        snow[1] += 5  # Vitesse de chute
        if snow[1] > frame.shape[0]:
            snow[1] = 0  # Réinitialiser la position en haut
        cv2.circle(frame, (snow[0], snow[1]), 3, (255, 255, 255), -1)
    return frame

# Initialiser les flocons
num_snowflakes = 50
snowflakes = [[np.random.randint(0, 640), np.random.randint(0, 480)] for _ in range(num_snowflakes)]

# Fonction pour changer le fond
def change_background(frame, background_image):
    mask = cv2.inRange(frame, (200, 200, 200), (255, 255, 255))  # Détection du fond blanc
    frame[mask != 0] = background_image[mask != 0]
    return frame

# Menu interactif
def display_menu():
    print("=== Menu ===")
    print("1: Activer filtre sépia")
    print("2: Incruster bonnet et lunettes")
    print("3: Ajouter flocons de neige")
    print("4: Changer le fond")
    print("5: Quitter")

# Flux vidéo de la webcam
cap = cv2.VideoCapture(0)

apply_sepia_filter = False
add_overlays = False
add_snow = False
replace_bg = False

# Menu d'options
while True:
    display_menu()
    choice = input("Entrez votre choix : ")

    if choice == '1':
        apply_sepia_filter = True
    elif choice == '2':
        add_overlays = True
    elif choice == '3':
        add_snow = True
    elif choice == '4':
        replace_bg = True
    elif choice == '5':
        break

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Appliquer filtre sépia
    if apply_sepia_filter:
        frame = apply_sepia(frame)

    # Détection de visage et incrustation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        if add_overlays:
            # Redimensionner et incruster le bonnet et les lunettes
            bonnet_resized = cv2.resize(bonnet, (w, int(0.6 * h)))
            lunettes_resized = cv2.resize(lunettes, (w, int(0.25 * h)))
            frame = overlay_image(frame, bonnet_resized, x, y - int(0.6 * h))
            frame = overlay_image(frame, lunettes_resized, x, y + int(0.3 * h))

    # Ajouter les flocons de neige
    if add_snow:
        frame = add_snowflakes(frame, snowflakes)

    # Changer le fond
    if replace_bg and background_image is not None:
        background_resized = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))
        frame = change_background(frame, background_resized)

    # Afficher le résultat
    cv2.imshow("Traitement d'images", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

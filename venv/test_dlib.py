try:
    import dlib
    print("dlib a été importé avec succès !")
    print(f"Version de dlib : {dlib.__version__}")
except ImportError:
    print("Erreur : dlib n'a pas pu être importé.")
except Exception as e:
    print(f"Une erreur inattendue est survenue lors de l'importation de dlib : {e}")

try:
    import cv2
    print("OpenCV a été importé avec succès !")
    print(f"Version d'OpenCV : {cv2.__version__}")
except ImportError:
    print("Erreur : OpenCV n'a pas pu être importé.")
except Exception as e:
    print(f"Une erreur inattendue est survenue lors de l'importation d'OpenCV : {e}")

print("Votre environnement est prêt !")
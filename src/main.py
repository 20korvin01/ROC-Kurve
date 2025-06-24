import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import os
import tqdm









if __name__ == "__main__":
    # Verzeichnisse für die Bilder
    dir_cars = "./data/cars/"
    dir_street = "./data/street/"
    
    # Laden der Graustufenbilder
    # AUTOS
    cars = []
    cars_central = []
    cars_outside_central = []
    for filename in tqdm.tqdm(os.listdir(dir_cars), desc="Lade Fahrzeugbilder und zentrale Pixel"):
        gray_img = io.imread(os.path.join(dir_cars, filename))
        # Vollständige Graustufenbilder
        cars.append(gray_img.flatten())
        # Zentrale 3x3 Pixel extrahieren
        central_pixel = gray_img[gray_img.shape[0]//2-1:gray_img.shape[0]//2+2, gray_img.shape[1]//2-1:gray_img.shape[1]//2+2]
        cars_central.append(central_pixel.flatten())
        # Alle Pixel außer den zentralen 3x3 Pixel
        if gray_img.shape[0] > 3 and gray_img.shape[1] > 3:
            outside_central = np.delete(gray_img, slice(gray_img.shape[0]//2-1, gray_img.shape[0]//2+2), axis=0)
            outside_central = np.delete(outside_central, slice(gray_img.shape[1]//2-1, gray_img.shape[1]//2+2), axis=1)
            cars_outside_central.append(outside_central.flatten())
            
    # STRASSEN
    streets = []
    for filename in tqdm.tqdm(os.listdir(dir_street), desc="Lade Straßenbilder"):
        gray_img = io.imread(os.path.join(dir_street, filename))
        streets.append(gray_img.flatten())
            

    ### 1. Stellen Sie drei empirische Grauwertverteilungen für Fahrzeuge auf und visualisieren Sie diese:
    ## a) mit allen Pixeln der Bildausschnitte
    cars = np.concatenate(cars)
    plt.figure(figsize=(10, 5))
    plt.hist(cars, bins=256, density=True, alpha=0.5, color='blue', label='Cars - All Pixels')
    plt.title('Grauwertverteilung der Fahrzeuge (alle Pixel)')
    plt.xlabel('Grauwert')
    plt.ylabel('Dichte')
    plt.legend()
    plt.savefig('plots/cars_all_pixels.png')
    
    print("Grauwertverteilung der Fahrzeuge (alle Pixel) gespeichert als 'cars_all_pixels.png'")
    
    
    ## b) nur auf Basis der jeweils zentralen 3x3 Pixel eines jeden Bildausschnittes
    cars_central = np.concatenate(cars_central)
    plt.figure(figsize=(10, 5))
    plt.hist(cars_central, bins=256, density=True, alpha=0.5, color='orange', label='Cars - Central 3x3 Pixels')
    plt.title('Grauwertverteilung der Fahrzeuge (zentrale 3x3 Pixel)')
    plt.xlabel('Grauwert')
    plt.ylabel('Dichte')
    plt.legend()
    plt.savefig('plots/cars_central.png')
    
    print("Grauwertverteilung der Fahrzeuge (zentrale 3x3 Pixel) gespeichert als 'cars_central.png'")
    
    
    ## c) nur auf Basis der außerhalb der zentralen 3x3 Pixel liegenden Pixel
    cars_outside_central = np.concatenate(cars_outside_central)
    plt.figure(figsize=(10, 5))
    plt.hist(cars_outside_central, bins=256, density=True, alpha=0.5, color='green', label='Cars - Outside Central 3x3 Pixels')
    plt.title('Grauwertverteilung der Fahrzeuge (außerhalb zentrale 3x3 Pixel)')
    plt.xlabel('Grauwert')
    plt.ylabel('Dichte')
    plt.legend()
    plt.savefig('plots/cars_outside_central.png')
    
    print("Grauwertverteilung der Fahrzeuge (außerhalb zentrale 3x3 Pixel) gespeichert als 'cars_outside_central.png'")
    
    
    # Bonus --> Alle drei Verteilungen in einem Plot
    plt.figure(figsize=(10, 5))
    plt.hist(cars, bins=256, density=True, alpha=0.5, color='blue', label='Cars - All Pixels')
    plt.hist(cars_central, bins=256, density=True, alpha=0.5, color='orange', label='Cars - Central 3x3 Pixels')
    plt.hist(cars_outside_central, bins=256, density=True, alpha=0.5, color='green', label='Cars - Outside Central 3x3 Pixels')
    plt.title('Grauwertverteilung der Fahrzeuge (alle Pixel, zentrale 3x3 Pixel, außerhalb zentrale 3x3 Pixel)')
    plt.xlabel('Grauwert')
    plt.ylabel('Dichte')
    plt.legend()
    plt.savefig('plots/cars_all_distributions.png')
    
    print("Grauwertverteilung der Fahrzeuge (alle Verteilungen) gespeichert als 'cars_all_distributions.png'")


    ### 2. Interpretieren Sie die Verteilungen. Weshalb unterscheiden sich diese?

    ### 3. Stellen Sie die empirische Grauwertverteilung der Straßenflächen auf und visualisieren Sie diese.
    streets = np.concatenate(streets)
    plt.figure(figsize=(10, 5))
    plt.hist(streets, bins=256, density=True, alpha=0.5, color='purple', label='Streets')
    plt.title('Grauwertverteilung der Straßenflächen')
    plt.xlabel('Grauwert')
    plt.ylabel('Dichte')
    plt.legend()
    plt.savefig('plots/streets_distribution.png')
    
    print("Grauwertverteilung der Straßenflächen gespeichert als 'streets_distribution.png'")

    ### 4. Erstellen Sie iterativ durch Verschieben des Detektionsschwellwertes die ROC-Kurve aus den Verteilungen 1a mit 3 sowie 1b mit 3

    ### 5. Wie erklären Sie sich anhand der ROC-Kurve das Detektorverhalten? Welche Verbesserungen könnte man am Detektor anbringen, um die Detektionsperformance zu verbessern?

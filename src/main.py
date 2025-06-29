import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import label
from skimage import io
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
    print(len(cars), "Pixel in allen Fahrzeugbildern")
    # plt.figure(figsize=(10, 5))
    # plt.hist(cars, bins=np.arange(257)-0.5, density=True, alpha=0.5, color='blue', label='Cars - All Pixels')
    # plt.ylim(0, 0.02)
    # plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}".replace('.', ',')))
    # plt.xlabel('Grauwert')
    # plt.ylabel('Dichte')
    # plt.tight_layout()
    # plt.savefig('plots/cars_all_pixels.png')
    
    # print("Grauwertverteilung der Fahrzeuge (alle Pixel) gespeichert als 'cars_all_pixels.png'")
    
    ## b) nur auf Basis der jeweils zentralen 3x3 Pixel eines jeden Bildausschnittes
    cars_central = np.concatenate(cars_central)
    # plt.figure(figsize=(10, 5))
    # plt.hist(cars_central, bins=np.arange(257)-0.5, density=True, alpha=0.5, color='orange', label='Cars - Central 3x3 Pixels')
    # plt.ylim(0, 0.02)
    # plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}".replace('.', ',')))
    # plt.xlabel('Grauwert')
    # plt.ylabel('Dichte')
    # plt.tight_layout()
    # plt.savefig('plots/cars_central.png')
    
    # print("Grauwertverteilung der Fahrzeuge (zentrale 3x3 Pixel) gespeichert als 'cars_central.png'")
    
    
    ## c) nur auf Basis der außerhalb der zentralen 3x3 Pixel liegenden Pixel
    cars_outside_central = np.concatenate(cars_outside_central)
    # plt.figure(figsize=(10, 5))
    # plt.hist(cars_outside_central, bins=np.arange(257)-0.5, density=True, alpha=0.5, color='green', label='Cars - Outside Central 3x3 Pixels')
    # plt.ylim(0, 0.02)
    # plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}".replace('.', ',')))
    # plt.xlabel('Grauwert')
    # plt.ylabel('Dichte')
    # plt.tight_layout()
    # plt.savefig('plots/cars_outside_central.png')
    
    # print("Grauwertverteilung der Fahrzeuge (außerhalb zentrale 3x3 Pixel) gespeichert als 'cars_outside_central.png'")
    
    
    # Bonus --> Alle drei Verteilungen in einem Plot
    # plt.figure(figsize=(10, 5))
    # plt.hist(cars, bins=np.arange(257)-0.5, density=True, alpha=0.5, color='blue', label='Alle Pixel')
    # plt.hist(cars_central, bins=np.arange(257)-0.5, density=True, alpha=0.5, color='orange', label='Zentrale 3x3 Pixel')
    # plt.hist(cars_outside_central, bins=np.arange(257)-0.5, density=True, alpha=0.5, color='green', label='Außerhalb zentrale 3x3 Pixel')
    # plt.ylim(0, 0.02)
    # plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}".replace('.', ',')))
    # plt.xlabel('Grauwert')
    # plt.ylabel('Dichte')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('plots/cars_all_distributions.png')
    
    # print("Grauwertverteilung der Fahrzeuge (alle Verteilungen) gespeichert als 'cars_all_distributions.png'")


    ### 2. Interpretieren Sie die Verteilungen. Weshalb unterscheiden sich diese?

    ### 3. Stellen Sie die empirische Grauwertverteilung der Straßenflächen auf und visualisieren Sie diese.
    streets = np.concatenate(streets)
    # plt.figure(figsize=(10, 5))
    # plt.hist(streets, bins=np.arange(257)-0.5, density=True, alpha=0.5, color='purple', label='Streets')
    # plt.ylim(0, 0.02)
    # plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}".replace('.', ',')))
    # plt.xlabel('Grauwert')
    # plt.ylabel('Dichte')
    # plt.tight_layout()
    # plt.savefig('plots/streets_distribution.png')
    
    # print("Grauwertverteilung der Straßenflächen gespeichert als 'streets_distribution.png'")

    ### 4. Erstellen Sie iterativ durch Verschieben des Detektionsschwellwertes die ROC-Kurve aus den Verteilungen 1a mit 3 sowie 1b mit 3
    
    # Verteilungen 1a (alle Pixel der Fahrzeugbilder) und 3 (Straßenbilder)
    true_positive_rates_1a3 = []
    false_positive_rates_1a3 = []
      
    for threshold in range(256):
        # True Positives (TP) und False Positives (FP) zählen
        tp = np.sum(cars <= threshold)
        fn = np.sum(cars > threshold)
        
        fp = np.sum(streets <= threshold)
        tn = np.sum(streets > threshold)
        
        # Berechnung der Raten
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        
        true_positive_rates_1a3.append(tpr)
        false_positive_rates_1a3.append(fpr)

    np.savetxt('rocPointsTaskA.txt', np.column_stack((np.round(np.array(false_positive_rates_1a3), 3),
                                            np.round(np.array(true_positive_rates_1a3), 3),
                                            np.arange(0, 256, 1))), delimiter=';', header='fpr,tpr,threshold', fmt='%.3f')

    plt.figure(figsize=(10, 5))
    plt.scatter(false_positive_rates_1a3, true_positive_rates_1a3, label='ROC-Kurve (All cars vs Streets)')
    plt.scatter(0,0,color='red',label='ausgewählte Schwellwerte') # dummy-plot für übersichtliche legende
    for i in range(0, len(false_positive_rates_1a3), 20):
        plt.text(false_positive_rates_1a3[i], true_positive_rates_1a3[i]+0.02, str(i), fontsize=8, ha='right', va='bottom')
        plt.scatter(false_positive_rates_1a3[i], true_positive_rates_1a3[i],color='red')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Zufallsrate')
    plt.title('ROC-Kurve für gesamte Fahrzeugbilder gegen Straßenbilder')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/roc_curve_all_cars_vs_streets.png')
    
    print("ROC-Kurve für gesamte Fahrzeugbilder gegen Straßenbilder gespeichert als 'roc_curve_all_cars_vs_streets.png'")
    
    # Verteilungen 1b (zentrale 3x3 Pixel der Fahrzeugbilder) und 3 (Straßenbilder)
    true_positive_rates_1b3 = []
    false_positive_rates_1b3 = []
    
    for threshold in range(256):
        # True Positives (TP) und False Positives (FP) zählen
        tp = np.sum(cars_central <= threshold)
        fn = np.sum(cars_central > threshold)
        
        fp = np.sum(streets <= threshold)
        tn = np.sum(streets > threshold)
        
        # Berechnung der Raten
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        
        true_positive_rates_1b3.append(tpr)
        false_positive_rates_1b3.append(fpr)

    np.savetxt('rocPointsTaskB.txt',np.column_stack((np.round(np.array(false_positive_rates_1b3),3),
                                            np.round(np.array(true_positive_rates_1b3),3),np.arange(0,256,1)))
                                            ,delimiter=';',header='fpr,tpr,threshold',fmt='%.3f')


    plt.figure(figsize=(10, 5))
    plt.scatter(false_positive_rates_1b3, true_positive_rates_1b3, label='ROC-Kurve (Central cars vs Streets)')
    plt.scatter(0,0,color='red',label='ausgewählte Schwellwerte') # dummy-plot für übersichtliche legende
    for i in range(0, len(false_positive_rates_1b3), 20):
        plt.text(false_positive_rates_1b3[i], true_positive_rates_1b3[i]+0.02, str(i), fontsize=8, ha='right', va='bottom')
        plt.scatter(false_positive_rates_1b3[i], true_positive_rates_1b3[i],color='red')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Zufallsrate')
    plt.title('ROC-Kurve für zentrale 3x3 Pixel der Fahrzeugbilder gegen Straßenbilder')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/roc_curve_central_cars_vs_streets.png')
    
    print("ROC-Kurve für zentrale 3x3 Pixel der Fahrzeugbilder gegen Straßenbilder gespeichert als 'roc_curve_central_cars_vs_streets.png'")

    ### 5. Wie erklären Sie sich anhand der ROC-Kurve das Detektorverhalten? Welche Verbesserungen könnte man am Detektor anbringen, um die Detektionsperformance zu verbessern?

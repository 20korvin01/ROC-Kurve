# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 09:27:11 2023

@author: juliu
"""

import numpy as np
#import cv2
import glob
from PIL import Image
import matplotlib.pyplot as plt

cars = 'data/cars'

cars_gesamt = []
cars_innen = []
cars_aussen = []

# Iteriere über alle PNG-Dateien im Verzeichnis
for datei in glob.glob(cars + '/*.png'):
    bild = Image.open(datei)
    bild_array_ganz = np.array(bild)
    bild_array_innen =bild_array_ganz[7:9, 7:9]
    vektor_ganz = bild_array_ganz.flatten().tolist()
    vektor_innen = bild_array_innen.flatten().tolist()
    cars_gesamt = cars_gesamt + vektor_ganz
    cars_innen = cars_innen + vektor_innen

#%%

cars_gesamt_np = np.array(cars_gesamt)
cars_innen_np = np.array(cars_innen)

histogram_gesamt, bins1 = np.histogram(cars_gesamt_np.flatten(),bins=256, range=[0,255])
histogram_innen, bins2 = np.histogram(cars_innen_np.flatten(),bins=256, range=[0,255])
histogram_aussen=histogram_gesamt-histogram_innen

plt.figure()
plt.bar(range(256),histogram_gesamt)
plt.title('Histogramm Auto gesamt')
plt.show()
plt.close()

plt.figure()
plt.bar(range(256),histogram_innen)
plt.title('Histogramm Auto 3x3')
plt.show()
plt.close()

plt.figure()
plt.bar(range(256),histogram_aussen)
plt.title('Histogramm Auto aussen')
plt.show()
plt.close()

#%%
street = 'data/street'

street_gesamt = []

# Iteriere über alle PNG-Dateien im Verzeichnis
for datei in glob.glob(street + '/*.png'):
    bild = Image.open(datei)
    bild_array_ganz = np.array(bild)
    vektor_ganz = bild_array_ganz.flatten().tolist()
    street_gesamt = street_gesamt + vektor_ganz

#%%

street_gesamt_np = np.array(street_gesamt)

histogram_gesamt, bins1 = np.histogram(street_gesamt_np.flatten(),bins=256, range=[0,255])

plt.figure()
plt.bar(range(256),histogram_gesamt)
plt.title('Histogramm Straße gesamt')
plt.show()
plt.close()


#%% Ganzes Bild mit Straße
tpr= np.zeros((256,1))
fpr= np.zeros((256,1))
for S in range(256):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for c in cars_gesamt:
        if c<=S:
            tp = tp + 1 
        else:
            fn = fn + 1
            
    for s in street_gesamt:
        if s<=S:
            fp = fp + 1
        else:
            tn = tn + 1
    tpr[S] = tp/(tp+fn)
    fpr[S] = fp/(fp+tn)
    
#%%
plt.figure()
plt.plot(fpr,tpr)
plt.title("ROC-Kurve für ganzes Bild Auto und Straße")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.axis('tight')
plt.show()
plt.close()

#%% Bildausschnitt mit Straße
tpr= np.zeros((256,1))
fpr= np.zeros((256,1))
for S in range(256):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for c in cars_innen:
        if c<=S:
            tp = tp + 1 
        else:
            fn = fn + 1
            
    for s in street_gesamt:
        if s<=S:
            fp = fp + 1
        else:
            tn = tn + 1
    tpr[S] = tp/(tp+fn)
    fpr[S] = fp/(fp+tn)
    
#%%
plt.figure()
plt.plot(fpr,tpr)
plt.title("ROC-Kurve für Bildausschnitt Auto und Straße")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.axis('tight')  
plt.show()
plt.close()

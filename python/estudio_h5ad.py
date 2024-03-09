# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 19:27:42 2024

@author: Carla
"""

import h5py

# Especifica la ruta del archivo .h5ad
# archivo_h5ad = "C:/Users/Carla/Desktop/4o/Q1/TFG/python/microbiome_data_clr.h5ad"
archivo_h5ad = "C:/Users/Carla/Desktop/4o/Q1/TFG/output_clr.h5ad"

# Abre el archivo HDF5 en modo de lectura
with h5py.File(archivo_h5ad, "r") as archivo_hdf5:
    # Imprime las claves principales (grupos) en el archivo HDF5
    print("Claves principales en el archivo HDF5:", list(archivo_hdf5.keys()))

    # Accede a la matriz principal (.X) y observa sus primeras filas
    matriz_principal = archivo_hdf5["X"][:]
    print("Matriz principal (.X):", matriz_principal)

    # Accede a los metadatos de observaciones (.obs) y observa algunas claves
    metadatos_obs = archivo_hdf5["obs"]
    print("Metadatos de observaciones (.obs) - Claves:", list(metadatos_obs.keys()))

    # Accede a los metadatos de variables (.var) y observa algunas claves
    metadatos_var = archivo_hdf5["var"]
    print("Metadatos de variables (.var) - Claves:", list(metadatos_var.keys()))
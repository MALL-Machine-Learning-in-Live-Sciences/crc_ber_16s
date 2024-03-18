# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:34:31 2024

@author: Carla
"""

import pandas as pd
from anndata import AnnData, concat

# Lista de rutas a las muestras
# sample_data_files = ["C:/Users/Carla/Desktop/4o/Q1/TFG/samples/sample_data_1.csv", 
#                       "C:/Users/Carla/Desktop/4o/Q1/TFG/samples/sample_data_2.csv",
#                       "C:/Users/Carla/Desktop/4o/Q1/TFG/samples/sample_data_3.csv",
#                       "C:/Users/Carla/Desktop/4o/Q1/TFG/samples/sample_data_4.csv",
#                       "C:/Users/Carla/Desktop/4o/Q1/TFG/samples/sample_data_5.csv",
#                       "C:/Users/Carla/Desktop/4o/Q1/TFG/samples/sample_data_6.csv",
#                       "C:/Users/Carla/Desktop/4o/Q1/TFG/samples/sample_data_7.csv"]
# otu_table_files = ["C:/Users/Carla/Desktop/4o/Q1/TFG/samples/otu_table_1.csv",
#                     "C:/Users/Carla/Desktop/4o/Q1/TFG/samples/otu_table_2.csv",
#                     "C:/Users/Carla/Desktop/4o/Q1/TFG/samples/otu_table_3.csv",
#                     "C:/Users/Carla/Desktop/4o/Q1/TFG/samples/otu_table_4.csv",
#                     "C:/Users/Carla/Desktop/4o/Q1/TFG/samples/otu_table_5.csv",
#                     "C:/Users/Carla/Desktop/4o/Q1/TFG/samples/otu_table_6.csv",
#                     "C:/Users/Carla/Desktop/4o/Q1/TFG/samples/otu_table_7.csv"]

sample_data_files = ["C:/Users/Carla/Desktop/4o/Q1/TFG/samples/sample_data_abr.csv"]
otu_table_files = ["C:/Users/Carla/Desktop/4o/Q1/TFG/samples/otu_table_abr.csv"]


# Lista para almacenar objetos AnnData
adata_list = []

# Iterar sobre los archivos de otu_table y cargar los datos en objetos AnnData
for otu_file, sample_file in zip(otu_table_files, sample_data_files):
    # Cargar los datos de otu_table y sample_data en pandas DataFrames
    otu_table = pd.read_csv(otu_file, index_col=0)
    sample_data = pd.read_csv(sample_file, index_col=0)
    
    # Crear un objeto AnnData
    adata_cohort = AnnData(X=otu_table.values.T, obs=sample_data)
    
    # Renombrar la columna 'cohort_id' a 'data_source'
    adata_cohort.obs['data_source'] = sample_data['project']  # Usar los valores de la columna 'project' como 'data_source'
    
    # Asegurarse de que todos los objetos AnnData tengan el atributo .X asignado
    if 'X' not in adata_cohort.layers.keys():
        adata_cohort.layers['X'] = adata_cohort.X

    # Agregar el objeto AnnData a la lista
    adata_list.append(adata_cohort)

# Concatenar los objetos AnnData en la lista y crear el objeto
ad = concat(adata_list)
ad.write_h5ad("microbiome_data_abr.h5ad")
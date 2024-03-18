# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:09:14 2024

@author: Carla
"""

import os
import pandas as pd
import anndata

# Ruta al directorio donde deseas guardar los archivos CSV
output_directory = "C:/Users/Carla/Desktop/4o/Q1/TFG/samples/output"

# Cargar el archivo .h5ad
ad = anndata.read_h5ad("C:/Users/Carla/Desktop/4o/Q1/TFG/crc_ber_16s/python/output_abr.h5ad")
# ad_clr = anndata.read_h5ad("C:/Users/Carla/Desktop/4o/Q1/TFG/output_clr.h5ad")

otu_table_df = pd.DataFrame(ad.X, index=ad.obs.index, columns=ad.var.index)
sample_data_df = ad.obs

otu_table_df.to_csv(os.path.join(output_directory, "otu_table_abr.csv"))
sample_data_df.to_csv(os.path.join(output_directory, "sample_data_abr.csv"))


# # Iterar sobre las cohortes en el objeto AnnData
# for project in ad_clr.obs['data_source'].unique():
#     # Filtrar el objeto AnnData para obtener solo los datos de una cohorte específica
#     adata_cohort = ad_clr[ad.obs['data_source'] == project]
    
#     # Extraer la matriz de datos (X) y las anotaciones de observaciones (sample_data)
#     otu_table = pd.DataFrame(adata_cohort.X.T, index=adata_cohort.var_names, columns=adata_cohort.obs_names)
#     sample_data = adata_cohort.obs
    
#     # Escribir los datos en archivos CSV
#     otu_table_file = os.path.join(output_directory, f"otu_table_{project}_clr.csv")
#     sample_data_file = os.path.join(output_directory, f"sample_data_{project}_clr.csv")
    
#     otu_table.to_csv(otu_table_file)
#     sample_data.to_csv(sample_data_file)

#     print(f"Archivos CSV creados para la cohorte {project}: {otu_table_file}, {sample_data_file}")
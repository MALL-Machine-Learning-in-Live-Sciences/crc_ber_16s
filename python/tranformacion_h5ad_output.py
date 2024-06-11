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
ad = anndata.read_h5ad("C:/Users/Carla/Desktop/4o/Q1/TFG/crc_ber_16s/python/output_alr_filt.h5ad")

otu_table_df = pd.DataFrame(ad.X, index=ad.obs.index, columns=ad.var.index)
sample_data_df = ad.obs

otu_table_df.to_csv(os.path.join(output_directory, "otu_table_alr_filt.csv"))
sample_data_df.to_csv(os.path.join(output_directory, "sample_data_alr_filt.csv"))
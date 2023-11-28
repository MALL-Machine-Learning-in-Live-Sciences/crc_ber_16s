# crc_ber_16s
TFG - GCED

Eliminación de batch-effects en cohortes de microbioma 16S-rRNA mediante variational autoencoders para estudios de cáncer colorectal. 

Conseguiremos esto mediante el uso de MOBER.

## MOBER

MOBER (Multi Origin Batch Effect Remover) es un método basado en aprendizaje profundo que realiza una integración biológicamente relevante de perfiles de expresión génica de modelos preclínicos y tumores clínicos. MOBER puede utilizarse para orientar la selección de líneas celulares y xenoinjertos derivados de pacientes, así como para identificar modelos que se asemejen más estrechamente a los tumores clínicos.

A continuación se detallan los pasos a seguir para su ejecución.

### Prerrequisitos 
- Entorno python: asegúrate de tener instalado un entorno python en tu sistema. El código requiere que Python esté configurado y en funcionamiento. 
- Sistema de GPU con CUDA y PyTorch: asegúrate de tener un sistema con una GPU compatible y que CUDA esté instalado. Además,instala PyTorch de acuerdo con la versión de CUDA compatible.
- Instalación mober: clona el repositorio de la herramienta y realiza la instalación de MOBER. Puedes hacerlo mediante:
```bash
# Clonar el repositorio MOBER
git clone https://github.com/Novartis/mober.git

# Cambiar al directorio del repositorio clonado
cd mober

# Instalar MOBER 
pip install -e .

# Verificar la instalación de MOBER
mober --help
```

- Archivo de Entrada .h5ad para entrenamiento: el archivo de entrada para el entrenamiento de MOBER debe estar en formato AnnData y guardado como .h5ad. Código de ejemplo con matrices aleatorias:
```python
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

# Generar datos aleatorios
num_samples = 100
num_genes = 500

# Datos aleatorios
X = np.random.rand(num_samples, num_genes)

sample_ids = [f'S{i}' for i in range(1, num_samples + 1)]
data_sources = ['group_A' if i % 2 == 0 else 'group_B' for i in range(1, num_samples + 1)]
sampInfo = pd.DataFrame({'data_source': data_sources}, index=sample_ids)

gene_ids = [f'Gene{i}' for i in range(1, num_genes + 1)]
geneInfo = pd.DataFrame(index=gene_ids)

# Crear objeto AnnData
adata = sc.AnnData(csr_matrix(X), obs=sampInfo, var=geneInfo)

# Guardar en un archivo h5ad
adata.write('input.h5ad')
```
### Uso de la herramienta

#### Entrenamiento
```bash
mober train \
--train_file input.h5ad \
--output_dir ../tmp_data/test
```
En este caso, el modelo entrenado estará en ../tmp_data/test/models y las métricas de entrenamiento y los parámetros utilizados para el entrenamiento estarán en ../tmp_data/test/metrics, en formato TSV.

#### Proyección 

Una vez que el modelo está entrenado, la proyección se puede realizar de dos maneras diferentes:

1. A través de la línea de comandos
```bash
mober projection \
--model_dir path_to_where_models_and_metrics_folders_are/models \
--onto TCGA \  # debe ser uno de los IDs de batch utilizados en el entrenamiento
--projection_file input.h5ad \
--output_file outname.h5ad \
--decimals 4
```
2. A través de scripts de Python
```python
from mober.core.projection import load_model, do_projection
import scanpy as sc
import torch

model_dir = 'path_to_where_models_and_metrics_folders_are/models'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
adata = sc.read('projection_file.h5ad')
model, features, label_encode = load_model(model_dir, device)
adata = adata[:,features]

proj_adata, z_adata = do_projection(model,adata, onto, label_encode, device, batch_size=1600)
proj_adata.write('outname.h5ad')

# proj_adata -> contiene los valores proyectados
# z_adata -> contiene los embeddings de las muestras en el espacio latente
```
## Singularity

Utilizamos un entorno singularity para la ejecución de MOBER, con el objetivo de contar con un entorno "portátil" en el que se tenga todo lo necesario para la ejecución de la herramienta.  

A continuación se detallan los pasos a seguir para la creación del entorno.

### Prerrequisitos

### Creación

### Ejemplo de uso
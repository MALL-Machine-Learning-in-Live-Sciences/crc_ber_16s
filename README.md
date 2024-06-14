# crc_ber_16s
TFG - GCED

Eliminación de batch-effects en cohortes de microbioma 16S-rRNA mediante variational autoencoders para estudios de cáncer colorectal. 

Se estudia esa cuestión mediante el uso de MOBER.

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

### Aplicación

Hemos aplicado MOBER mediante los comandos mencionados, de la siguiente manera:

1. Entrenamiento 

```bash 
mober train --train_file "C:\Users\Carla\Desktop\4o\Q1\TFG\microbiome_data.h5ad" --output_dir "..\tmp_data\test"
```
Aquí:
- microbiome_data.h5ad es un objeto generado a partir de otro objeto en formato phyloseq, resultado a su vez del preprocesado de las cohortes de microbioma proporcionadas como datos base.
- La dirección del output es la definida por defecto en el modelo.

En cuanto al output obtenemos como resultado, por una parte, tres ficheros (train_loss_adv, train_loss_ae y train_loss_tot) con información sobre la pérdida obtenida durante el entrenamiento (adversarial, de reconstrucción y total); y por otra parte, varios ficheros csv con características y parámetros sobre los datos input y el entrenamiento, sumados a los modelos finales entrenados para las dos componentes claves de MOBER: el codificador automático variacional condicional (batch_ae_final.model) y la red neuronal discriminadora de fuente (src_adv_final.model).

2. Proyección 

```bash
mober projection 
--model_dir "C:\Users\Carla\Desktop\4o\Q1\TFG\mober\tmp_data\test\models"
--onto (1-7)
--projection_file "C:\Users\Carla\Desktop\4o\Q1\TFG\microbiome_data.h5ad"
--output_file "C:\Users\Carla\Desktop\4o\Q1\TFG\output.h5ad"
--decimals 4
```
En este caso:
- La dirección del modelo es la definida por defecto hacia donde se encuentran los modelos comentados en el anterior apartado.
- onto: una de nuestras cohortes sobre la que las muestras serán proyectadas.
- El fichero sobre el que haremos proyección será el utilizado para el entrenamiento, en nuestro caso microbiome_data.h5ad.
- El output es un nuevo fichero .h5ad, que contendrá los valores proyectados, y con los que trabajaremos en próximmos pasos.
- decimals representa al número de decimales para el fichero generado como output. Valor 4 por defecto.

## Singularity

Singularity es una herramienta que permite la creación, ejecución y manejo de contenedores.

Utilizamos un entorno singularity para la ejecución de un contenedor portable de las funciones desarrolladas, encapsulando el entorno y las dependencias del proyecto, a través de un contenedor.

A continuación se detallan tanto los pasos seguidos para la creación del contenedor como los comandos necesarios para su ejecución.

### Prerrequisitos

- Se recomienda utilizar Singularity en un entorno Linux, pero no es un requisito indispensable. 

#### Instalación 
- Instalación de dependencias
```bash
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev uuid-dev libgpgme11-dev squashfs-tools libseccomp-dev wget pkg-config git cryptsetup
```
- Descarga y compilado
```bash
VERSION= 3.9.4  # Por lo general usar la versión más reciente disponible
wget https://github.com/sylabs/singularity/releases/download/v$VERSION/singularity-$VERSION.tar.gz
tar -xzf singularity-$VERSION.tar.gz
cd singularity-$VERSION
./mconfig
make -C ./builddir
sudo make -C ./builddir install
```
- Verificación de la instalación
```bash
singularity --version
```
### Creación de un entorno 
Una manera sencilla de trabajar dentro de Singularity, y la que utilizaremos, es la generación de una imagen tipo sif con la que podremos interactuar. 

Para el montaje de esta imagen principal, crearemos un archivo de texto que define cómo construir una imagen de contenedor Singularity (llamado "recipe"). En nuestro caso, podría ser:
```bash
Bootstrap: docker
From: python:3.8

%labels
    Maintainer TuNombre
    Version 1.0

%post
    # Actualizar e instalar dependencias necesarias
    apt-get update && apt-get install -y \
        git \
        && rm -rf /var/lib/apt/lists/*

    # Clonar el repositorio de mober
    git clone https://github.com/Novartis/mober.git /opt/mober

    # Instalar mober
    pip install --upgrade pip
    pip install torch
    pip install -e /opt/mober
    pip install numexpr

%runscript
    exec "$@"

%startscript
    exec "$@"
```

Una vez creado el recipe (recipe.txt), montamos la imagen:
```bash
$ sudo singularity build mober_container.sif mober_container.def
```
siendo mober_container.sif la imagen a crear y mober_container.def el recipe mostrado

Una vez montado el contenedor, ya se puede ejecutar MOBER dentro de él:

1. Entrenamiento 
```bash
singularity run mober_container.sif mober train \
--train_file data/microbiome_data.h5ad \
--output_dir . 
```

2. Proyección 
```bash
singularity run mober_container.sif mober projection \
--model_dir models \
--onto Zackular \
--projection_file data/microbiome_data.h5ad \
--output_file projection/output.h5ad \
--decimals 4
```


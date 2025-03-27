# Loki
Building on OmiCLIP, a visual–omics foundation model designed to bridge omics data and hematoxylin and eosin (H&E) images, we developed the **Loki** platform, which has five key functions: tissue alignment using ST or H&E images, cell type decomposition of ST or H&E images using scRNA-seq as a reference, tissue annotation of ST or H&E images based on bulk RNA-seq or marker genes, ST gene expression prediction from H&E images, and histology image–transcriptomics retrieval.

This README provides a quick overview of how to set up and use Loki.



## User Manual and Notebooks
You can view the Loki website and notebooks [here](https://guangyuwanglab2021.github.io/Loki/).



## Source Code
All source code for Loki is contained in the `./src/loki` directory.



## Installation (It takes about 5 mins to finish the installation on MacBook Pro)

1. **Create a Conda environment**:
   ```bash
   conda create -n loki_env python=3.9
   conda activate loki_env
   ```

2. **Navigate to the Loki source directory and install Loki**:
   ```bash
   cd ./src
   pip install .
   ```


## Usage
Once Loki is installed, you can import it in your Python scripts or notebooks:
   ```python
   import loki.preprocess
   import loki.utils
   import loki.plot

   import loki.align
   import loki.annotate
   import loki.decompose
   import loki.retrieve
   import loki.predex
   ```



## STbank
The ST-bank database are avaliable from [Google Drive link](https://drive.google.com/drive/folders/1J15cO-pXTwkTjRAR-v-_nQkqXNfcCNn3?usp=sharing).

The links_to_raw_data.xlsx file includes the source paper names, doi links, and download links of the raw data.
The text.csv file includes the gene sentences with paired image patches.
The image.tar.gz includes the image patches.



## Pretrained weights
The pretrained weights are avaliable on [Hugging Face](https://huggingface.co/WangGuangyuLab/Loki).



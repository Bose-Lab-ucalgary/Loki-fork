This directory contains the scripts to run the downstream analysis of patient cancer risk prediction.

Step-by-Step Guide
------------------

1. **Download the data files**
   
   This includes the whole slide images (WSIs) of The Cancer Genome Atlas (TCGA) from the Genomic Data Commons (GDC) `data portal <https://portal.gdc.cancer.gov/>`_, and the bulk RNA-seq data, clinical data from the `cBioPortal <https://www.cbioportal.org/>`_. Arrange the data files int the following structure as shown in the `data_source <data_source>`_ directory.

2. **Prepare the input files**

   a. Extract the gene expression data from the bulk RNA-seq data by keeping only top-ranking gene names.
   
   b. Create WSI tiles using CLAM library. Please refer to the `CLAM repo <https://github.com/mahmoodlab/CLAM/>`_ for more details.

   c. Generate the input metadata files. The files contain case_id, clinical data and the associated wsi paths. Run in command line as follows::

       python 0.1_prepare_datacsv.py

   Files are then saved in the `dataset_csv <dataset_csv>`_ directory.

3. **Calculate the transcriptomic embedding**

   This is done using the pretrained model (e.g. OmiCLIP) and the tile-level image embeddings. Run in command line as follows::

       python calculate_text_embeddings.py

4. **Calculate the tile embeddings**

   This is done using the pretrained model (e.g. OmiCLIP) and the tile-level image embeddings. Run in command line as follows::

       python calculate_patch_image_embeddings.py

5. **Calculate the tile weigthts**

   This is based on the cosine similarity between the transcriptomic embedding and the tile-level image embeddings. Then the patient-level embedding is calculated by aggregating the tile-level embeddings weighted by the tile attentions. Run::

       python 0.2_calculate_patch_weights.py

6. **Split the patient-level embeddings and the clinical data**

   This is done into training and test sets using a 5-fold cross-validation method. Make sure to keep the ratio of the censored and uncensored patients in the training and test sets the same. Run in command line as follows::

       python 0.3_create_splits.py

7. **Train the patient cancer risk prediction model**

   This is done using the patient-level embeddings and the clinical data, with a simple fully connected network as shown in the `survival_model.py <survival_model.py>`_ file. Run in command line as follows::

       python 0.4_run_surv.py

   It can also be run in a Jupyter notebook. An example run for TCGA low-grade glioma (LGG) dataset is shown in the Jupyter notebook `run_survival_analysis.ipynb <0.4_run_survival_analysis.ipynb>`_.

8. **Evaluate the model performance**

   Finally, evaluate the performance by fitting the predicted survival with Kaplan-Meier curves and running log-rank tests for statistical significance. Functions are available in the `0.4_run_surv.py <0.4_run_surv.py>`_ file.
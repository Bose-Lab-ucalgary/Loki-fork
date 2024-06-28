Instructions
============

STEP 1. Prepare the input scRNA-seq data files and the ground truth cell type annotation
----------------------------------------------------------------------------------------

.. code-block:: bash

    sh ./01.run_prepare.sh

Prepare the input files
^^^^^^^^^^^^^^^^^^^^^^^

Raw scRNA-seq data files in h5ad format saved in the `data_source <./data_source/>`_ directory. 

.. code-block:: bash

    data_source/
    ├── housekeeping_genes.csv
    └── SC_BC.h5ad

Output files
^^^^^^^^^^^^

.. code-block:: bash

    SC_BC/
    ├── SC_BC_labels.csv
    └── SC_BC_top100_names.npy

1. CSV file containing the cell type labels. 
2. Numpy array containing the top-ranking gene names. 

STEP 2. Calculate the transcriptomic embedding using the pretrained model (e.g. OmiCLIP)
----------------------------------------------------------------------------------------

.. code-block:: bash

    sbatch ./02.run_calculate_embedding.sh

Calculate the transcriptomic embedding using the pretrained model. The example script was run on a SLURM cluster with
GPUs. 

Output files
^^^^^^^^^^^^

.. code-block:: bash

    embeddings_omiclip/
    └── omiclip-SC_BC_top100_names-text_embeddings.pt

PyTorch tensor containing the transcriptomic embeddings.

STEP 3. Train the cell type classifier. In this case, we use a simple bayesian classifier
-----------------------------------------------------------------------------------------

.. code-block:: bash

    sbatch ./03.run_classifier.slurm

Train the cell type classifier using the transcriptomic embeddings and randomly sampled fractions of labeled cells.

Output files
^^^^^^^^^^^^

.. code-block:: bash

    results/
    ├── meta
    │   └── SC_BC
    │       ├── omiclip-SC_BC-text-confusion_matrix_1000_run1.csv
    │       ├── omiclip-SC_BC-text-confusion_matrix_1000_run2.csv
    │       ├── omiclip-SC_BC-text-confusion_matrix_1000_run3.csv
    │       ├── omiclip-SC_BC-text-confusion_matrix_1000_run4.csv
    │       ├── omiclip-SC_BC-text-confusion_matrix_1000_run5.csv
    │       ├── omiclip-SC_BC-text-confusion_matrix_1000_run6.csv
    │       ├── omiclip-SC_BC-text-confusion_matrix_1000_run7.csv
    │       ├── omiclip-SC_BC-text-confusion_matrix_1000_run8.csv
    │       ├── omiclip-SC_BC-text-confusion_matrix_1000_run9.csv
    │       └── omiclip-SC_BC-text-confusion_matrix_1000_run10.csv
    ├── omiclip-SC_BC-text-confusion_matrix_1000.csv
    └── omiclip-SC_BC-text-f1_scores.csv

Here only parital files are displayed (using 1000, 3% labelled data for training and the rest for validation, randomly
pick 10 times). The confusion matrices and F1 scores for 10 runs. The output files can be used to make plots (refer to
the Jupyter notebook  `make_plots.ipynb <./make_plots.ipynb>`_).  

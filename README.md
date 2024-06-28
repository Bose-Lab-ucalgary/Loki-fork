OmiCLIP
===========
## A visual–omics foundation model to bridge histopathology image with transcriptomics

**Abstract:** Computational pathology has emerged as a powerful tool for revolutionizing routine pathology through artificial intelligence (AI)-driven analysis of pathology images. Recent advancements in omics technologies, such as single-cell RNA sequencing (scRNA-seq), spatial transcriptomics (ST), and proteomics, have enriched the field by providing detailed genomic information alongside tissue histology. However, existing computational models primarily emphasize single modality development, either omics-based analysis or image-based analysis, leaving a gap for jointly supporting genomic and histopathology analysis for computational pathology. To address this gap, we developed OmiCLIP, a visual–omics foundation model to bridge omics and hematoxylin and eosin (H&E) images using tissue patches of Visium data. In the transcriptomics domain, we generated a ‘sentence’ representing the transcriptomics of a tissue patch, which concatenates gene symbols of top highly expressed genes. In total, we collected 2.2 million paired tissue images and transcriptomics data including 32 organ types. We used this large-scale set of histology image–transcriptomics pairs to finetune a CLIP-based foundation model for pathology, incorporating both image and transcriptomics data. We systematically evaluated OmiCLIP on six tasks by 14 independent validation datasets and 8 in-house patient tissues. For zero-shot tissue classification, OmiCLIP achieved F1 scores of 0.96–0.59 in four independent datasets, surpassing OpenAI CLIP’s F1 scores of 0.34–0.03. We also incorporated OmiCLIP and PLIP, a state-of-the-art visual–language foundation model, and largely enhanced the performance of tissue annotation. Next, we applied OmiCLIP to predict patient risk in six cancer types based on whole-slide images (WSIs) and RNA sequencing data of biopsies. Our analysis shows OmiCLIP’s image and transcriptomic embeddings could leverage the survival prediction of cancer patients. For cell type annotation, we evaluated the few-shot performance by training a linear classifier using OmiCLIP embeddings, achieving 0.89–0.77 F1 scores trained on 3% labeled cells. These capabilities represent a fundamental step toward bridging and applying foundation models in genomics for histopathology.


## Installation
First clone the repo and cd into the directory:
```shell
git clone https://github.com/GuangyuWangLab/OmiCLIP.git
cd OmiCLIP
```
Then create a conda env and install the dependencies:
```shell
conda create -n omiclip python=3.9 -y
conda activate omiclip
pip install --upgrade pip
pip install -e .
```

## Preparing and loading the model
1. Download the model weights from the Huggingface model page (weights will be avaliable from Hugging Face after manuscript accepted). 

First create the `checkpoints` directory inside the root of the repo:
```shell
mkdir -p checkpoints/omiclip/
```
Then download the pretrained model (`pytorch_model.bin`) and place it in the `OmiCLIP/checkpoints/omiclip/` directory. 

2. Loading the model

```python
from open_clip import create_model_from_pretrained
model, preprocess = create_model_from_pretrained("coca_ViT-L-14", checkpoint_path="checkpoints/omiclip/pytorch_model.bin")
```

## Overview of specific usages
Refer to the notebooks below for detailed examples.

### Basic usage of the model:
See [**here**](notebooks/basic_usage.ipynb).

### Zeroshot classification:
See [**here**](notebooks/zeroshot_classification.ipynb).

### Zeroshot cross-modality retrieval (image-to-text):
See [**here**](notebooks/zeroshot_retrieval.ipynb).

### Patient cancer risk prediction:
See [**here**](scripts/cancer_patient_survival_tcga).

### Few-shot single cell annotation:
See [**here**](scripts/few_shot_single_cell_annotation).


## Acknowledgements
The project was built on top of the amazing repository [openclip](https://github.com/mlfoundations/open_clip) for model training. We thank the authors and developers for their contribution. 

## License and Terms of Use
ⓒ GuangyuWang Lab. This model and associated code are released under the [CC-BY-NC-ND 4.0]((https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en)) license and may only be used for non-commercial, academic research purposes with proper attribution. 

# main function
# use arguments to specify the pretrained model
# output path
# `test case`
# usage: python calculate_embeddings.py --model biomedclip --output_path /condo/wanglab/tmhpxz9/wc-coca/embeddings --test_case DigestPath
import argparse
parser = argparse.ArgumentParser(description='Calculate embeddings for test cases')
parser.add_argument('--model', type=str, default='biomedclip', help='pretrained model')
parser.add_argument('--output_path', type=str, default='text_embeddings', help='output path')
parser.add_argument('--test_case', type=str, default='SC_BC', help='test case')
parser.add_argument('--clip_context', action='store_true', help='force tokenizer context length to be 76')
parser.add_argument('--use_gpu', action='store_true', help='use gpu') 
args = parser.parse_args()

import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
from glob import glob
from pathlib import Path
from tqdm import tqdm

from open_clip import create_model_from_pretrained, get_tokenizer

# set device
DEVICE = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
print("Device: ", DEVICE)

def get_image_embedding(query_PIL_image, model_pretrained, image_preprocess):
    image_input = torch.stack([image_preprocess(query_PIL_image)]).to(DEVICE)
    with torch.no_grad():
        query_image_embedding_n = model_pretrained.encode_image(image_input, normalize=True).cpu().detach()

    return query_image_embedding_n

def get_image_embeddings(query_PIL_images, model_pretrained, image_preprocess):
    image_inputs = torch.stack([image_preprocess(img) for img in query_PIL_images]).to(DEVICE)
    with torch.no_grad():
        query_image_embeddings_n = model_pretrained.encode_image(image_inputs, normalize=True).cpu().detach()

    return query_image_embeddings_n

def get_image_embeddings_large(query_PIL_images, model_pretrained, image_preprocess, batch_size=100, image_paths=False):
    from tqdm import tqdm
    n = len(query_PIL_images)

    if n <= batch_size:
        return get_image_embeddings(query_PIL_images, model_pretrained, image_preprocess)

    query_image_embeddings_n = []

    for i in tqdm(range(0, n, batch_size)):
        batch_images = query_PIL_images[i:i+batch_size]
        if image_paths:
            batch_images = [Image.open(img_path) for img_path in batch_images]
        image_embeddings_n = get_image_embeddings(batch_images, model_pretrained, image_preprocess)
        query_image_embeddings_n.append(image_embeddings_n)

    return torch.cat(query_image_embeddings_n, dim=0)

def get_text_embeddings(query_texts, model_pretrained, tokenizer, context_length=None):
    """
    query_texts: list of strings

    Returns:
    query_text_embedding_n: torch.Tensor
    """
    # Warning: this function is not memory efficient
    # somehow the GPU memory required > 40 GB for more than 3000 samples

    if context_length is not None:
        query_input = tokenizer(query_texts, context_length=context_length).to(DEVICE)
    else:
        query_input = tokenizer(query_texts).to(DEVICE)

    with torch.no_grad():
        query_text_embedding_n = model_pretrained.encode_text(query_input, normalize=True).cpu().detach()

    return query_text_embedding_n

def get_text_embeddings_large(query_texts, model_pretrained, tokenizer, batch_size=100, context_length=None):
    n = len(query_texts)

    if n <= batch_size:
        return get_text_embeddings(query_texts, model_pretrained, tokenizer, context_length=context_length)

    query_text_embeddings_n = []

    for i in tqdm(range(0, n, batch_size)):
        batch_texts = query_texts[i:i+batch_size]
        text_embeddings_n = get_text_embeddings(batch_texts, model_pretrained, tokenizer)
        query_text_embeddings_n.append(text_embeddings_n)

    return torch.cat(query_text_embeddings_n, dim=0)

def get_image_path(df, column_name='img_path', root_path="/condo/wanglab/shared/wxc/clip/data"):
    return [os.path.join(root_path, row[column_name]) for index, row in df.iterrows()]

def get_label(df, column_name='label'):
    """
    return a list of labels
    """
    return df[column_name].tolist()

def read_text_file(file_path, sep=' '):
    text_array = np.load(file_path, allow_pickle=True)
    cell_text_vector = np.apply_along_axis(lambda x: sep.join(x), 1, text_array)
    return cell_text_vector

#### Load model
# change to directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Current working directory: {os.getcwd()}")


PRETRAINED_NETS = {'biomedclip': 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                   'coca': 'coca_ViT-L-14',
                   'openaiclip': 'ViT-L-14-336',
                   'omiclip': 'coca_ViT-L-14'}

MODEL_WEIGHTS_ROOT = "/condo/wanglab/tmhpxz9/wc-coca/model_weights"
model_paths = {'biomedclip': None,
               'openaiclip': "openai",
               'coca': "mscoco_finetuned_laion2B-s13B-b90k",
               'omiclip': os.path.join(MODEL_WEIGHTS_ROOT, "omiclip.pt")
               }

PRETRAINED_NET = PRETRAINED_NETS[args.model]
model_path = model_paths[args.model]
output_path = args.output_path
test_case = args.test_case
model = args.model

if model in ['biomedclip', 'omiclip', 'coca', 'openaiclip']:
    tokenizer = get_tokenizer(PRETRAINED_NET)
else:
    raise ValueError(f"Model {model} not supported")

if args.clip_context:
    context_length = 76
else:
    context_length = None

batch_size = 100 if DEVICE == "cpu" else 1000

os.makedirs(output_path, exist_ok=True)
model_pretrained, preprocess = create_model_from_pretrained(
        PRETRAINED_NET,
        device = DEVICE,
        pretrained=model_path
        )

if os.path.exists(test_case):
    samples_meta = glob(f"{test_case}/{test_case}*top100_names.npy")
    for sample in samples_meta:
        sample_name = Path(sample).stem
        labels = read_text_file(sample)
        text_embeddings = get_text_embeddings_large(labels, model_pretrained, tokenizer, batch_size=batch_size, context_length=context_length)
        torch.save(text_embeddings, os.path.join(output_path, f"{model}-{sample_name}-text_embeddings.pt"))
else:
    print(f"Test case {test_case} not found")
    exit(1)

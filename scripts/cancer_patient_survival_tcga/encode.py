PRETRAINED_NETS = {
    'biomedclip':
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
    'coca':
        'coca_ViT-L-14',
    'openaiclip':
        'ViT-L-14-336',
    'omiclip':
        'coca_ViT-L-14'
}

MODEL_WEIGHTS_PATH = "/condo/wanglab/tmhpxz9/wc-coca/model_weights/omiclip.pt"
model_paths = {
    'biomedclip': None,
    'openaiclip': "openai",
    'coca': "mscoco_finetuned_laion2B-s13B-b90k",
    'omiclip': MODEL_WEIGHTS_PATH
}


def get_image_embedding(
    query_PIL_image, model_pretrained, image_preprocess, DEVICE="cuda"
):
    import torch

    image_input = torch.stack([image_preprocess(query_PIL_image)]).to(DEVICE)
    with torch.no_grad():
        query_image_embedding_n = model_pretrained.encode_image(
            image_input, normalize=True
        ).cpu().detach()

    return query_image_embedding_n


def get_image_embeddings(
    query_PIL_images, model_pretrained, image_preprocess, DEVICE="cuda"
):
    import torch

    #    print("Using device:", DEVICE)

    image_inputs = torch.stack(
        [image_preprocess(img) for img in query_PIL_images]
    ).to(DEVICE)
    with torch.no_grad():
        query_image_embeddings_n = model_pretrained.encode_image(
            image_inputs, normalize=True
        ).cpu().detach()

    return query_image_embeddings_n


def get_image_embeddings_large(
    query_PIL_images,
    model_pretrained,
    image_preprocess,
    batch_size=100,
    image_paths=False,
    DEVICE="cuda"
):
    """
    query_PIL_images: list of PIL images or list of image paths depending on the value of image_paths

    return a torch.Tensor of image embeddings
    """
    import torch
    from PIL import Image
    from tqdm import tqdm

    #    print("Using device:", DEVICE)

    n = len(query_PIL_images)
    if n <= batch_size:
        return get_image_embeddings(
            query_PIL_images, model_pretrained, image_preprocess, DEVICE=DEVICE
        )

    query_image_embeddings_n = []

    # instead of report per batch, we report per image
    pbar = tqdm(total=n, desc="Encoding images", unit="image")

    for i in range(0, n, batch_size):
        batch_images = query_PIL_images[i:i + batch_size]
        if image_paths:
            batch_images = [Image.open(img_path) for img_path in batch_images]
        image_embeddings_n = get_image_embeddings(
            batch_images, model_pretrained, image_preprocess, DEVICE=DEVICE
        )
        query_image_embeddings_n.append(image_embeddings_n)
        pbar.update(len(batch_images))

    pbar.close()

    return torch.cat(query_image_embeddings_n, dim=0)


def get_text_embeddings(
    query_texts,
    model_pretrained,
    tokenizer,
    context_length=None,
    DEVICE="cuda"
):
    """
    query_texts: list of strings

    Returns:
    query_text_embedding_n: torch.Tensor
    """
    # Warning: this function is not memory efficient
    # somehow the GPU memory required > 40 GB for more than 3000 samples
    import torch

    #    print("Context length in batch:", context_length)
    if context_length is not None:
        query_input = tokenizer(query_texts,
                                context_length=context_length).to(DEVICE)
    else:
        query_input = tokenizer(query_texts).to(DEVICE)


#    print("Query input shape:", query_input.shape)
    with torch.no_grad():
        query_text_embedding_n = model_pretrained.encode_text(
            query_input, normalize=True
        ).cpu().detach()

    return query_text_embedding_n


def get_text_embeddings_large(
    query_texts,
    model_pretrained,
    tokenizer,
    batch_size=100,
    context_length=None,
    DEVICE="cuda"
):
    import torch
    from tqdm import tqdm

    #    print("Context length:", context_length)

    n = len(query_texts)

    if n <= batch_size:
        return get_text_embeddings(
            query_texts,
            model_pretrained,
            tokenizer,
            context_length=context_length
        )

    query_text_embeddings_n = []

    for i in tqdm(range(0, n, batch_size)):
        batch_texts = query_texts[i:i + batch_size]
        text_embeddings_n = get_text_embeddings(
            batch_texts,
            model_pretrained,
            tokenizer,
            context_length=context_length,
            DEVICE=DEVICE
        )
        query_text_embeddings_n.append(text_embeddings_n)

    return torch.cat(query_text_embeddings_n, dim=0)


def get_image_path(
    df,
    column_name='img_path',
    root_path="/condo/wanglab/shared/wxc/clip/data"
):
    import os

    return [
        os.path.join(root_path, row[column_name])
        for index, row in df.iterrows()
    ]


def get_label(df, column_name='label'):
    """
    return a list of labels
    """
    return df[column_name].tolist()


def read_text_file(file_path, sep=' '):
    import numpy as np

    text_array = np.load(file_path, allow_pickle=True)
    cell_text_vector = np.apply_along_axis(lambda x: sep.join(x), 1, text_array)
    return cell_text_vector

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import json
import cv2
from sklearn.decomposition import PCA
from open_clip import create_model_from_pretrained, get_tokenizer



def load_model(model_path, device):
    """
    Loads a pretrained CoCa (CLIP-like) model, along with its preprocessing function and tokenizer, 
    using the specified model checkpoint.

    :param model_path: File path or URL to the pretrained model checkpoint. This is passed to 
                       `create_model_from_pretrained` as the `pretrained` argument.
    :type model_path: str
    :param device: The device on which to load the model (e.g., 'cpu' or 'cuda').
    :type device: str or torch.device
    :return: A tuple `(model, preprocess, tokenizer)` where:
             - model: The loaded CoCa model.
             - preprocess: A function or transform that preprocesses input data for the model.
             - tokenizer: A tokenizer appropriate for textual input to the model.
    :rtype: (nn.Module, callable, callable)
    """
    # Create the model and its preprocessing transform from the specified checkpoint
    model, preprocess = create_model_from_pretrained(
        "coca_ViT-L-14", device=device, pretrained=model_path
    )
    
    # Retrieve a tokenizer compatible with the "coca_ViT-L-14" architecture
    tokenizer = get_tokenizer('coca_ViT-L-14')

    return model, preprocess, tokenizer



def encode_image(model, preprocess, image):
    """
    Encodes an image into a normalized feature embedding using the specified model and preprocessing function.

    :param model: A model object that provides an `encode_image` method (e.g., a CLIP or CoCa model).
    :type model: torch.nn.Module
    :param preprocess: A preprocessing function that transforms the input image into a tensor 
                       suitable for the model. Typically something returning a PyTorch tensor.
    :type preprocess: callable
    :param image: The input image (PIL Image, NumPy array, or other format supported by `preprocess`).
    :type image: PIL.Image.Image or numpy.ndarray
    :return: A single normalized image embedding as a PyTorch tensor of shape (1, embedding_dim).
    :rtype: torch.Tensor
    """
    # Preprocess the image, then stack to create a batch of size 1
    image_input = torch.stack([preprocess(image)])

    # Generate the image features without gradient tracking
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    # Normalize embeddings across the feature dimension (L2 normalization)
    image_embeddings = F.normalize(image_features, p=2, dim=-1)

    return image_embeddings



def encode_image_patches(model, preprocess, data_dir, img_list):
    """
    Encodes multiple image patches into normalized feature embeddings using a specified model and preprocess function.
    
    :param model: A model object that provides an `encode_image` method (e.g., a CLIP or CoCa model).
    :type model: torch.nn.Module
    :param preprocess: A preprocessing function that transforms the input image into a tensor 
                       suitable for the model. Typically something returning a PyTorch tensor.
    :type preprocess: callable
    :param data_dir: The base directory containing image data.
    :type data_dir: str
    :param img_list: A list of image filenames (strings). Each filename corresponds to a patch image 
                     stored in `data_dir/demo_data/patch/`.
    :type img_list: list[str]
    :return: A PyTorch tensor of shape (N, 1, embedding_dim), containing the normalized embeddings 
             for each image in `img_list`.
    :rtype: torch.Tensor
    """

    # Prepare a list to hold each image's feature embedding
    image_embeddings = []

    # Loop through each image name in the provided list
    for img_name in img_list:
        # Build the path to the patch image and open it
        image_path = os.path.join(data_dir, 'demo_data', 'patch', img_name)
        image = Image.open(image_path)

        # Encode the image using the model & preprocess; returns shape (1, embedding_dim)
        image_features = encode_image(model, preprocess, image)

        # Accumulate the feature embeddings in the list
        image_embeddings.append(image_features)

    # Convert the list of embeddings to a NumPy array, then to a PyTorch tensor
    # Resulting shape will be (N, 1, embedding_dim)
    image_embeddings = torch.from_numpy(np.array(image_embeddings))

    # Normalize all embeddings across the feature dimension (L2 normalization)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

    return image_embeddings



def encode_text(model, tokenizer, text):
    """
    Encodes text into a normalized feature embedding using a specified model and tokenizer.

    :param model: A model object that provides an `encode_text` method (e.g., a CLIP-like or CoCa model).
    :type model: torch.nn.Module
    :param tokenizer: A tokenizer function that converts the input text into a format suitable for `model.encode_text`.
                      Typically returns token IDs, attention masks, etc. as a torch.Tensor or similar structure.
    :type tokenizer: callable
    :param text: The input text (string or list of strings) to be encoded.
    :type text: str or list[str]
    :return: A PyTorch tensor of shape (batch_size, embedding_dim) containing the L2-normalized text embeddings.
    :rtype: torch.Tensor
    """

    # Convert text to the appropriate tokenized representation
    text_input = tokenizer(text)

    # Run the model in no-grad mode (not tracking gradients, saving memory and compute)
    with torch.no_grad():
        text_features = model.encode_text(text_input)

    # Normalize embeddings to unit length
    text_embeddings = F.normalize(text_features, p=2, dim=-1)

    return text_embeddings



def encode_text_df(model, tokenizer, df, col_name):
    """
    Encodes text from a specified column in a pandas DataFrame using the given model and tokenizer,
    returning a PyTorch tensor of normalized text embeddings.

    :param model: A model object that provides an `encode_text` method (e.g., a CLIP-like or CoCa model).
    :type model: torch.nn.Module
    :param tokenizer: A tokenizer function that converts the input text into a format suitable for `model.encode_text`.
    :type tokenizer: callable
    :param df: A pandas DataFrame from which text will be extracted.
    :type df: pandas.DataFrame
    :param col_name: The name of the column in `df` that contains the text to be encoded.
    :type col_name: str
    :return: A PyTorch tensor containing the L2-normalized text embeddings, 
             where the shape is (number_of_rows, embedding_dim).
    :rtype: torch.Tensor
    """

    # Prepare a list to hold each row's text embedding
    text_embeddings = []

    # Loop through each index in the DataFrame
    for idx in df.index:
        # Retrieve text from the specified column for the current row
        text = df[df.index == idx][col_name][0]

        # Encode the text using the provided model and tokenizer
        text_features = encode_text(model, tokenizer, text)

        # Accumulate the embedding tensor
        text_embeddings.append(text_features)

    # Convert the list of embeddings (likely shape [N, embedding_dim]) into a NumPy array, then to a torch tensor
    text_embeddings = torch.from_numpy(np.array(text_embeddings))

    # Normalize embeddings to unit length across the feature dimension
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    return text_embeddings



def get_pca_by_fit(tar_features, src_features):
    """
    Applies PCA to target features and transforms both target and source features using the fitted PCA model.
    Combines the PCA-transformed features from both target and source datasets and returns the combined data 
    along with batch labels indicating the origin of each sample.

    :param tar_features: Numpy array of target features (samples by features).
    :param src_features: Numpy array of source features (samples by features).
    :return: 
        - pca_comb_features: A numpy array containing PCA-transformed target and source features combined.
        - pca_comb_features_batch: A numpy array of batch labels indicating which samples are from target (0) and source (1).
    """

    pca = PCA(n_components=3)
    
    # Fit the PCA model on the target features (transposed to fit on features)
    pca_fit_tar = pca.fit(tar_features.T)
    
    # Transform the target and source features using the fitted PCA model
    pca_tar = pca_fit_tar.transform(tar_features.T)  # Transform target features
    pca_src = pca_fit_tar.transform(src_features.T)  # Transform source features using the same PCA fit
    
    # Combine the PCA-transformed target and source features
    pca_comb_features = np.concatenate((pca_tar, pca_src))
    
    # Create a batch label array: 0 for target features, 1 for source features
    pca_comb_features_batch = np.array([0] * len(pca_tar) + [1] * len(pca_src))

    return pca_comb_features, pca_comb_features_batch



def cap_quantile(weight, cap_max=None, cap_min=None):
    """
    Caps the values in the 'weight' array based on the specified quantile thresholds for maximum and minimum values.
    If the quantile thresholds are provided, the function will replace values above or below these thresholds 
    with the corresponding quantile values.

    :param weight: Numpy array of weights to be capped.
    :param cap_max: Quantile threshold for the maximum cap. Values above this quantile will be capped. 
                    If None, no maximum capping will be applied.
    :param cap_min: Quantile threshold for the minimum cap. Values below this quantile will be capped. 
                    If None, no minimum capping will be applied.
    :return: Numpy array with the values capped at the specified quantiles.
    """
    
    # If a maximum cap is specified, calculate the value at the specified cap_max quantile
    if cap_max is not None:
        cap_max = np.quantile(weight, cap_max)  # Get the value at the cap_max quantile
    
    # If a minimum cap is specified, calculate the value at the specified cap_min quantile
    if cap_min is not None:
        cap_min = np.quantile(weight, cap_min)  # Get the value at the cap_min quantile
    
    # Cap the values in 'weight' array to not exceed the maximum cap (cap_max)
    weight = np.minimum(weight, cap_max)
    
    # Cap the values in 'weight' array to not go below the minimum cap (cap_min)
    weight = np.maximum(weight, cap_min)
    
    return weight



def read_polygons(file_path, slide_id):
    """
    Reads polygon data from a JSON file for a specific slide ID, extracting coordinates, colors, and thickness.

    :param file_path: Path to the JSON file containing polygon configurations.
    :param slide_id: Identifier for the specific slide whose polygon data is to be extracted.
    :return: 
        - polygons: A list of numpy arrays, where each array contains the coordinates of a polygon.
        - polygon_colors: A list of color values corresponding to each polygon.
        - polygon_thickness: A list of thickness values for each polygon's border.
    """

    # Open the JSON file and load the polygon configurations into a Python dictionary
    with open(file_path, 'r') as f:
        polygons_configs = json.load(f)

    # Check if the given slide_id exists in the polygon configurations
    if slide_id not in polygons_configs:
        return None, None, None  # If slide_id is not found, return None for all outputs

    # Extract the polygon coordinates, colors, and thicknesses for the given slide_id
    polygons = [np.array(poly['coords']) for poly in polygons_configs[slide_id]]  # Convert polygon coordinates to numpy arrays
    polygon_colors = [poly['color'] for poly in polygons_configs[slide_id]]  # Extract the color for each polygon
    polygon_thickness = [poly['thickness'] for poly in polygons_configs[slide_id]]  # Extract the thickness for each polygon

    # Return the polygons, their colors, and their thicknesses
    return polygons, polygon_colors, polygon_thickness



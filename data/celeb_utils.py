
import sys
import json
import torch
from typing import List
from pathlib import Path
from time import perf_counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image

from llm_requests import get_embedding

@dataclass
class CelebMeta:
    """
    Contains metadata for the CelebA dataset.

    Args:
        img_height (int): Height of the images in the dataset.
        img_width (int): Width of the images in the dataset.
        dataset_base_path (Path): Path to the base directory of the dataset.
        images_pth (Path): Path to the directory containing the images.
        partitn_inf_pth (Path): Path to the file containing the partition information.
        attrib_inf_pth (Path): Path to the file containing the attribute information.
        normalize_mean (tuple): Mean of the normalization values for each attribute.
        normalize_std (tuple): Standard deviation of the normalization values for each attribute.
    """
    orig_img_height: int = 218
    orig_img_width: int = 178
    img_height: int = 208
    img_width: int = 178
    dataset_base_path: Path = None
    images_pth: Path = Path('Img/img_align_celeba/')
    partitn_inf_pth: Path = Path('Eval/list_eval_partition.txt')
    attrib_inf_pth: Path = Path('Anno/list_attr_celeba.txt')
    captions_path: Path = Path('captions.json')
    normalize_mean: tuple = (0.5061, 0.4254, 0.3828)
    normalize_std: tuple = (0.3105, 0.2903, 0.2896)
    

class CelebDataset(Dataset):
    def __init__(self, file_list: List[str], attrib_df: pd.DataFrame):
        """
        Args:
            file_list (List[str]): List of file paths to celeb face images.
            attrib_df (pd.DataFrame): Dataframe containing attributes for each image.
        """
        self.file_list = file_list
        self.attrib_df = attrib_df
        self.attrib_names = attrib_df.columns[1:].tolist()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop((208, 176)),
            transforms.Resize((224, 192)),
            transforms.Normalize(mean=CelebMeta.normalize_mean,
                                 std=CelebMeta.normalize_std)],
                                 )
    
    def get_normalize_stats(self, ):
        return dict(mean=list(CelebMeta.normalize_mean),
                    std=list(CelebMeta.normalize_std))

    def __getitem__(self, index: int):
        image = Image.open(str(self.file_list[index]))
        image = self.transforms(image)
        attributes = torch.tensor(self.attrib_df.iloc[index, 1:])
        attributes = torch.clip(attributes, 0, 1)
        return image, attributes

    def __len__(self):
        return len(self.file_list)


def get_dataset(data_path, split='train'):
    """
    Get celebA pytorch dataset.

    Args:
        data_path (str): The path to the data directory.
        split (str, optional): The split of the data to be used.

    Returns:
        CelebDataset: The dataset object.

    """
    partitn = get_data_partition(data_path)
    if split == 'test':
        file_list = partitn['test']
    else:
        file_list = np.hstack([partitn['train'], partitn['val']])
    
    attributes_data = get_attributes(data_path)
    attrib_df = attributes_data['attributes']
    attrib_names = attributes_data['attrib_names']
    attrib_df = attrib_df[attrib_df.fname.isin(file_list)]

    attrib_df.reset_index(inplace=True, drop=True)
    for col in attrib_names:
        attrib_df.loc[:, col] = attrib_df[col].apply(pd.to_numeric)

    prepend_base_path = lambda x: data_path / CelebMeta.images_pth / x
    file_list = list(map(prepend_base_path, attrib_df.fname.tolist()))
    
    print(f"Found {len(attrib_df)} samples in {split} split")
    return CelebDataset(file_list, attrib_df)



def get_data_partition(data_path):
    df = pd.read_csv(Path(data_path) / CelebMeta.partitn_inf_pth, 
                    delimiter= ' ', header=0, names=['fname', 'split'])
    val_split = df['fname'][df.split == 1].values
    test_split = df['fname'][df.split == 2].values
    train_split = df['fname'][df.split == 0].values
    return {'train': train_split, 'val': val_split, 'test': test_split}


def get_attributes(data_path):
    with open(Path(data_path) / CelebMeta.attrib_inf_pth) as f:
        data = f.read()
    data = data.split('\n')
    attrib_names = data[1].split()    
    parsed_rows = []
    for row in data[2:]:
        parsed_rows.append(row.split())
    df = pd.DataFrame(data=parsed_rows, columns=['fname']+attrib_names)
    df.dropna(inplace=True)
    return {'attributes': df, 'attrib_names':attrib_names}


def get_normalization_params(data_path, use_test_split=True):
    partitn = get_data_partition()
    if use_test_split:
        files = np.hstack([partitn['train'], partitn['val'], partitn['test']])
    else:
        files = np.hstack([partitn['train'], partitn['val']])
    N = len(files)
    n_pixels = CelebMeta.img_height * CelebMeta.img_width
    sum = torch.tensor([0., 0., 0.], dtype=torch.float64)
    sq_sum = torch.tensor([0., 0., 0.], dtype=torch.float64)           
    tick = perf_counter()
    for ix, file in enumerate(files):
        img = read_image(str(Path(data_path) / CelebMeta.images_pth / file)) / 255
        sum += img.sum(dim=[1,2]) / n_pixels
        sq_sum += img.square().sum(dim=[1,2]) / n_pixels
        elapsed = perf_counter() - tick
        progress_bar(ix, N, 
                     text=f'Processing... | elapsed : {elapsed:.3f}s :: ')

    mean = sum / N
    var = (sq_sum / N) - torch.square(mean)

    return {'mean':mean, 'var':var, 'std':torch.sqrt(var)}

def get_weights_for_loss(attrib_df):
    '''
    Computes per class weights to go along with BCEWITHLOGITSLOSS.
    if a dataset contains 100 +ve & 300 -ve examples of a single class,
        then pos_weight for the class should be equal to 300/100 = 3
    '''
    pos_weights = []
    total_count = attrib_df.shape[0]
    for col in attrib_df.columns[1:]:
        pos_count = (attrib_df[col] == 1).sum()
        neg_count = total_count - pos_count
        pos_weight = neg_count / pos_count
        pos_weights.append(pos_weight)
    return pos_weights


def progress_bar(current, total, bar_length=50, text="Progress"):
    anitext = ['\\', '|', '/', '-']
    percent = float(current) / total
    abs = f"{{{current} / {total}}}"
    arrow = '|' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(arrow))
    text = '[' + anitext[(current % 4)] + '] ' + text
    sys.stdout.write("\r{0}: [{1}] {2}% {3}".format(text, arrow + spaces, int(round(percent * 100)), abs))
    sys.stdout.flush()


def combiner(x):
    s = ''
    for key in x.attribute_wise_captions.keys():
        s += key + ' : ' + str(x.attribute_wise_captions[key]) + '\n'
    s += 'overall summary : ' + str(x.overall_caption)
    return s

def get_caption_embedding(dataset_path: Path):
    with open(dataset_path / CelebMeta.captions_path) as f:
        data = json.load(f)
    df = pd.DataFrame(data).T
    df.reset_index(inplace=True)
    df.rename(columns={'index':'fname'}, inplace=True)

    batch_size = 512
    batch_count = len(df) // batch_size
    batch_remainder = len(df) % batch_size
    batch_count += 1 if batch_remainder > 0 else 0

    stacked_embeddings = []
    for ix in range(batch_count):
        start_ix = ix * batch_size
        end_ix = start_ix + batch_size
        sliced_df = df.iloc[start_ix:end_ix]
        combined = sliced_df.apply(combiner, axis=1)
        inputs = combined.values.tolist()
        embeddings = get_embedding(inputs)['embedding']
        stacked_embeddings.append(embeddings)
        progress_bar(ix+1, batch_count)
    
    df.drop(columns=['attribute_wise_captions', 'overall_caption'], inplace=True)
    df['embedding'] = np.concatenate(stacked_embeddings).tolist()
    return df

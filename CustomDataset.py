import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from transformers import BertTokenizer

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Create a PyTorch Dataset


class TextDataset(Dataset):
    """
    A custom PyTorch Dataset for loading and tokenizing text files for sentiment classification.
    This dataset expects a directory structure where text files are organized in subdirectories
    (e.g., 'pos' for positive and 'neg' for negative sentiment). The label is inferred from the
    presence of 'neg' in the file path.
    Attributes:
        sentences_path (List[Path]): List of paths to text files.
        tokenizer (Callable): Tokenizer to convert text to token ids.
        max_length (int): Maximum sequence length for tokenization.
    Args:
        dataset_path (Path): Path to the root directory containing text files in subdirectories.
        tokenizer (Callable): Tokenizer function or object with a callable interface.
        max_length (int, optional): Maximum length of tokenized sequences. Defaults to 128.
    Methods:
        __len__():
            Returns the number of samples in the dataset.
        __getitem__(idx):
            Retrieves the tokenized input and label for the sample at the given index.
    Returns:
        dict: A dictionary containing:
            - 'input_ids': Tokenized input as a tensor.
            - 'label': Sentiment label as a float tensor (0 for negative, 1 for positive).
    """

    def __init__(self, dataset_path: str, tokenizer, max_length=128):
        self.sentences_path = list(Path(dataset_path).glob('*/*.txt'))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences_path)

    def __getitem__(self, idx):
        label = 0 if "neg" in str(self.sentences_path[idx]) else 1
        text = ''
        with open(self.sentences_path[idx], 'r', encoding='utf-8') as f:
            text = f.read().strip()
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
        )
        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        return {
            'input_ids': encoding,
            'label': torch.tensor(label, dtype=float)
        }

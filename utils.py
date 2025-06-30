import torch
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer


def model_training(
        train_dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str

):
    """
    Trains a PyTorch model for one epoch using a BERT encoder for input feature extraction.
    Args:
        train_dataloader (torch.utils.data.DataLoader): DataLoader providing training batches.
        model (torch.nn.Module): The model to be trained, which takes BERT-encoded features as input.
        loss_fn (torch.nn.Module): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (str): Device identifier (e.g., 'cpu' or 'cuda') for computation.
    Returns:
        tuple: A tuple containing:
            - total_loss (float): The accumulated loss over all batches.
            - total_accuracy (int): The total number of correct predictions across all batches.
        """
    model.train()
    model.to(device)
    encoder_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    total_loss = 0
    total_accuracy = 0
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        non_encoded = batch['input_ids']
        input_ids = non_encoded['input_ids'].to(device)
        attention_mask = non_encoded['attention_mask'].to(device)
        with torch.no_grad():
            encoded = encoder_model(input_ids=input_ids,
                                    attention_mask=attention_mask)

        X = encoded.last_hidden_state
        y = batch['label']
        X, y = X.to(device), y.to(device)

        y_logits = model(X).squeeze(1)

        loss = loss_fn(y_logits, y)
        total_loss += loss.item()

        y_prob = torch.sigmoid(y_logits)
        y_pred = (y_prob >= 0.5).long()

        accuracy = (y_pred == y).sum().item()
        total_accuracy += accuracy

        loss.backward()

        optimizer.step()
    return total_loss, total_accuracy


def model_testing(

        test_dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        device: str

):
    """
    Evaluates a PyTorch model on a test dataset using a BERT encoder for input feature extraction.
    Args:
        test_dataloader (torch.utils.data.DataLoader): DataLoader providing test batches.
        model (torch.nn.Module): The model to be evaluated, which takes BERT-encoded features as input.
        loss_fn (torch.nn.Module): Loss function used for evaluation.
        device (str): Device identifier (e.g., 'cpu' or 'cuda') for computation.
    Returns:
        tuple: A tuple containing:
            - total_loss (float): The accumulated loss over all batches.
            - total_accuracy (int): The total number of correct predictions across all batches.
    """
    model.eval()
    model.to(device)
    encoder_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    total_loss = 0
    total_accuracy = 0
    with torch.inference_mode():
        for batch in tqdm(test_dataloader):
            non_encoded = batch['input_ids']
            input_ids = non_encoded['input_ids'].to(device)
            attention_mask = non_encoded['attention_mask'].to(device)

            encoded = encoder_model(input_ids=input_ids,
                                    attention_mask=attention_mask)

            X = encoded.last_hidden_state
            y = batch['label']
            X, y = X.to(device), y.to(device)

            y_logits = model(X).squeeze(1)

            loss = loss_fn(y_logits, y)
            total_loss += loss.item()

            y_prob = torch.sigmoid(y_logits)
            y_pred = (y_prob >= 0.5).long()

            accuracy = (y_pred == y).sum().item()
            total_accuracy += accuracy

    return total_loss, total_accuracy


def get_label(logit: torch.Tensor):
    """
    Converts a model output logit into a probability and sentiment label.
    Args:
        logit (torch.Tensor): The raw output logit from a model, typically of shape (1,) or scalar.
    Returns:
        Tuple[float, str]: A tuple containing:
            - The predicted probability (as a percentage, float between 0 and 100).
            - The predicted sentiment label ('pos' for positive if probability < 0.5, 'neg' for negative otherwise).
    Note:
        The function applies a sigmoid activation to the logit to obtain the probability.
    """
    pred_prob = torch.sigmoid(logit).item()
    if pred_prob < 0.5:
        pred_label = "neg"
        pred_prob = 1 - pred_prob
    else:
        pred_label = "pos"
    return pred_prob * 100, pred_label


def get_text_tensor(text: str | list = "This is a sentence", device="cpu"):
    """
    Converts input text into a BERT-based tensor representation.
    This function tokenizes the input text using the 'bert-base-uncased' tokenizer,
    encodes it into input IDs and attention masks, and passes it through the
    corresponding BERT model to obtain the last hidden state tensor.
    Args:
        text (str, optional): The input text to be converted. Defaults to "This is a sentence".
        device (str, optional): The device to run the tensor operations on ('cpu' or 'cuda'). Defaults to "cpu".
    Returns:
        torch.Tensor: The last hidden state tensor output from the BERT model, representing the input text.
    """

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(
        text, truncation=True, padding='max_length', max_length=512)
    encoding = {k: torch.tensor(v).unsqueeze(0).to(device)
                for k, v in encoding.items()}
    model = BertModel.from_pretrained('bert-base-uncased')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    X = output.last_hidden_state

    return X

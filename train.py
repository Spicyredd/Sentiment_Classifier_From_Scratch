import torch
from utils import model_training, model_testing
from CustomModel import TransformerEncoder
from CustomDataset import TextDataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader


def train(
        model: torch.utils.data.DataLoader,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: str
):
    """
        Trains and evaluates a PyTorch model for a specified number of epochs.
        Args:
            model (torch.nn.Module): The neural network model to be trained.
            train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            test_dataloader (torch.utils.data.DataLoader): DataLoader for the testing/validation dataset.
            loss_fn (torch.nn.Module): Loss function used for training.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            epochs (int): Number of training epochs.
            device (str): Device to run the training on ('cpu' or 'cuda').
        Returns:
            None
        During each epoch, this function:
            - Trains the model on the training dataset and computes training loss and accuracy.
            - Evaluates the model on the test dataset and computes test loss and accuracy.
            - Prints the loss and accuracy for both training and testing phases.
    """
    for epoch in range(epochs):
        print("-"*20)
        print(f"Epoch: {epoch+1}")
        print("-"*20)
        train_loss, train_accuracy = model_training(train_dataloader=train_dataloader,
                                                    model=model,
                                                    loss_fn=loss_fn,
                                                    optimizer=optimizer,
                                                    device=device)
        test_loss, test_accuracy = model_testing(test_dataloader=test_dataloader,
                                                 model=model,
                                                 loss_fn=loss_fn,
                                                 device=device)
        print(
            f"Train Loss: {train_loss} | Train Accuracy: {train_accuracy} | Test Loss: {test_loss} | Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    model = TransformerEncoder(
        d_model=768,  # Dimension of model
        num_heads=4,  # No. of head to split into
        num_layers=2,  # No. of Encoder Block
        num_classes=1,  # No. of classes `1` for binary classification
        dropout_rate=0.1
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TextDataset(dataset_path="datasets/vipulgandhi/movie-review-dataset/versions/1/txt_sentoken",
                          tokenizer=tokenizer,
                          max_length=512)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [
                                                                0.8, 0.2])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=1e-03)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device :{device}")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    train(model=model,
          train_dataloader=train_dataloader,
          test_dataloader=test_dataloader,
          loss_fn=loss_fn,
          optimizer=optimizer,
          epochs=15,
          device=device)

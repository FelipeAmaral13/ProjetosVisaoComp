import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
from transformers import ViTModel, ViTImageProcessor
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler


class ViTFinetune(torch.nn.Module):
    def __init__(self, num_classes, pre_trained_model='google/vit-base-patch16-224-in21k', device='cuda'):
        """
        Initializes the ViTFinetune model.

        Parameters:
            num_classes (int): Number of classes for the image classification task.
            pre_trained_model (str): Pre-trained ViT model from Hugging Face Transformers.
                                     Default: 'google/vit-base-patch16-224-in21k'
        """
        super(ViTFinetune, self).__init__()
        self.vit = ViTModel.from_pretrained(pre_trained_model)
        self.feature_extractor = ViTImageProcessor.from_pretrained(
            pre_trained_model)
        self.classifier = torch.nn.Linear(
            self.vit.config.hidden_size, num_classes)
        self.device = device

    def forward(self, images):
        """
        Forward pass of the ViTFinetune model.

        Parameters:
            images (torch.Tensor): Batch of input images as a tensor.

        Returns:
            torch.Tensor: The model's predicted logits for the input images.
        """
        features = self.vit(images).last_hidden_state
        logits = self.classifier(features[:, 0])
        return logits

    def load_dataset(self, data_dir, batch_size, num_workers=4):
        """
        Load the training dataset for fine-tuning.

        Parameters:
            data_dir (str): Path to the directory containing the training dataset.
            batch_size (int): Batch size for training data loader.
            num_workers (int, optional): Number of subprocesses to use for data loading. Default: 4.

        Returns:
            DataLoader: Training data loader.
            int: Number of classes in the training dataset.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = ImageFolder(data_dir, transform=transform)
        num_classes = len(dataset.classes)
        self.classifier = torch.nn.Linear(
            self.vit.config.hidden_size, num_classes)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers)

        return train_loader, num_classes

    def load_validation_dataset(self, data_dir, batch_size, num_workers=4):
        """
        Load the validation dataset for evaluation during fine-tuning.

        Parameters:
            data_dir (str): Path to the directory containing the validation dataset.
            batch_size (int): Batch size for validation data loader.
            num_workers (int, optional): Number of subprocesses to use for data loading. Default: 4.

        Returns:
            DataLoader: Validation data loader.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        val_dataset = ImageFolder(data_dir, transform=transform)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers)

        return val_loader

    def set_device(self, device):
        """
        Set the device on which to perform inference (GPU or CPU).

        Parameters:
            device (str): 'cuda' for GPU, 'cpu' for CPU.
        """
        self.device = device
        self.to(device)

    def train_model(self, train_loader, val_loader, num_epochs, learning_rate=2e-5, patience=3):
        """
        Train the ViTFinetune model on the provided training dataset.

        Parameters:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            num_epochs (int): Number of epochs to train the model.
            learning_rate (float, optional): Learning rate for the optimizer. Default: 2e-5.
            patience (int, optional): Number of epochs to wait for improvement in validation loss before early stopping. Default: 3.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

        best_val_loss = float('inf')
        early_stop_counter = 0

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_loader.dataset.transform = train_transform

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0.0
            correct_train_predictions = 0
            total_train_samples = 0

            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                logits = self(images)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                _, predicted_labels = torch.max(logits, 1)
                correct_train_predictions += (predicted_labels ==
                                              labels).sum().item()
                total_train_samples += labels.size(0)

                if (i + 1) % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}] - Iteration [{i + 1}/{len(train_loader)}] - "
                          f"Loss: {loss.item():.4f}")

            train_accuracy = correct_train_predictions / total_train_samples
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {total_loss / len(train_loader):.4f} "
                  f"- Train Accuracy: {train_accuracy:.4f}")

            train_losses.append(total_loss / len(train_loader))
            train_accuracies.append(train_accuracy)

            self.eval()
            val_loss = 0.0
            correct_val_predictions = 0
            total_val_samples = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = self(images)
                    loss = loss_fn(logits, labels)
                    val_loss += loss.item()

                    _, predicted_labels = torch.max(logits, 1)
                    correct_val_predictions += (predicted_labels ==
                                                labels).sum().item()
                    total_val_samples += labels.size(0)

                val_accuracy = correct_val_predictions / total_val_samples
                print(f"Epoch [{epoch + 1}/{num_epochs}] - Validation Loss: {val_loss / len(val_loader):.4f} "
                      f"- Validation Accuracy: {val_accuracy:.4f}")

                val_losses.append(val_loss / len(val_loader))
                val_accuracies.append(val_accuracy)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter >= patience:
                    print(
                        f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
                    break

            scheduler.step()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1),
                 train_accuracies, label='Train Accuracy')
        plt.plot(range(1, num_epochs + 1), val_accuracies,
                 label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracies')
        plt.legend()
        plt.grid()
        plt.show()

        print("Training finished!")

    def save_model(self, save_path):
        """
        Save the trained model to a file.

        Parameters:
            save_path (str): File path to save the model state.
        """
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to '{save_path}'.")

    def load_model(self, model_path):
        """
        Load a trained model from a file.

        Parameters:
            model_path (str): File path of the saved model to load.
        """
        if os.path.isfile(model_path):
            self.load_state_dict(torch.load(model_path))
            print(f"Model loaded from '{model_path}'.")
        else:
            print(f"File '{model_path}' not found. The model was not loaded.")

    def predict_image(self, image_path):
        """
        Perform inference on a single image using the trained model.

        Parameters:
            image_path (str): File path of the input image.

        Returns:
            Tuple[int, np.ndarray, PIL.Image]: A tuple containing the predicted class label (int),
                                              predicted class probabilities (numpy array), and the input image (PIL.Image).
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        self.eval()

        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            logits = self(image_tensor)

            probabilities = torch.nn.functional.softmax(logits, dim=1)

            _, predicted_class = torch.max(probabilities, 1)

            return predicted_class.item(), probabilities[0].cpu().numpy(), image

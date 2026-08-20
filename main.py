from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vae import VAE
from pathlib import Path
from loss import vae_loss
import torch
import yaml


DEVICE ='cuda' if torch.cuda.is_available() else 'cpu'


def load_config():
    config_path = Path(__file__).resolve().parent / "config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def dataloading(config):

    transformer = transforms.ToTensor() # (28,28) -> (1,28,28)

    data_config = config["data"]
    SEED = config["seed"]
    BATCH_SIZE = data_config["batch_size"]
    ROOT = data_config["root"]
    TRAIN_SIZE = data_config["train_size"]
    VALIDATION_SIZE = data_config["validation_size"]

    full_train_dataset = datasets.FashionMNIST(
        root=data_config["root"],
        train=True,
        transform=transformer,
        download=True,
    )

    train_dataset = datasets.FashionMNIST(root=ROOT, train=True, transform=transformer, download=True)
    train_data, validation_data = torch.utils.data.random_split(
        train_dataset,
        [TRAIN_SIZE, VALIDATION_SIZE],
        generator=torch.Generator().manual_seed(SEED),
    )
    test_dataset = datasets.FashionMNIST(root=ROOT, train=False, transform=transformer)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, validation_loader, test_loader


def train(train_loader, validation_loader, nn_config):

    # TRAINING CONFIG
    training_config = nn_config["training"]
    LEARNING_RATE = float(training_config["learning_rate"])
    BETA_ONE = training_config["beta_one"]
    BETA_TWO = training_config["beta_two"]
    EPOCHS = training_config["epochs"]

    # SCHEDULER CONFIG
    scheduler_config = nn_config["scheduler"]
    LR_DECAY = scheduler_config["gamma"]
    STEP_SIZE = scheduler_config["step_size"]

    # LOSS CONFIG
    loss_config = nn_config["loss"]
    BETA = loss_config["beta"]

    model = VAE().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA_ONE, BETA_TWO))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY)

    training_losses = []
    validation_losses = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_reconstruction_loss = 0
        total_kld_loss = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            optimizer.zero_grad()
            reconstruction_logits, mu, log_var = model(images)

            loss, reconstruction_loss, kl_loss = vae_loss(images, reconstruction_logits, mu, log_var, beta=BETA)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_kld_loss += kl_loss.item() * images.size(0)
            total_reconstruction_loss += reconstruction_loss.item() * images.size(0)

        training_losses.append((total_loss / len(train_loader), total_kld_loss / len(train_loader), total_reconstruction_loss / len(train_loader)))
        validation_losses.append(validate(model, validation_loader, BETA))

        scheduler.step()

        torch.save(model, 'weights.pth')

        return training_losses, validation_losses


def validate(model, validation_loader, BETA):
    model.eval()
    total_loss = 0
    total_reconstruction_loss = 0
    total_kld_loss = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(DEVICE)
            reconstruction_logits, mu, log_var = model(images)

            loss, reconstruction_loss, kl_loss = vae_loss(reconstruction_logits, images, mu, log_var, beta=BETA)
            batch_size = reconstruction_logits.size(0)
            total_loss += reconstruction_loss.item() * batch_size
            total_reconstruction_loss += reconstruction_loss.item() * batch_size
            total_kld_loss += kl_loss.item() * batch_size

    return (
        total_loss / len(validation_loader),
        total_reconstruction_loss / len(validation_loader),
        total_kld_loss / len(validation_loader),
    )


if __name__ == "__main__":
    nn_config = load_config()
    train_set, val_set, test_set = dataloading(nn_config)
    training_loss, validation_loss = train(train_set, val_set, nn_config)
    print(training_loss[-1])
    print(validation_loss[-1])

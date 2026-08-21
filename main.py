from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vae import VAE
from pathlib import Path
from loss import vae_loss
from tqdm.auto import tqdm
import torch
import yaml
import pandas as pd


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

    datasets.FashionMNIST(
        root=ROOT,
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
    WEIGHT_DECAY = float(training_config["weight_decay"])

    # MODEL CONFIG
    model_config = nn_config["model"]
    LATENT_DIM = model_config["latent_dim"]

    # SCHEDULER CONFIG
    scheduler_config = nn_config["scheduler"]
    LR_DECAY = scheduler_config["gamma"]
    STEP_SIZE = scheduler_config["step_size"]

    # LOSS CONFIG
    loss_config = nn_config["loss"]
    BETA = loss_config["beta"]

    model = VAE(input_dim=LATENT_DIM).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(BETA_ONE, BETA_TWO), weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY)

    training_losses = []
    validation_losses = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_reconstruction_loss = 0
        total_kld_loss = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            unit="batch",
            dynamic_ncols=True,
            colour="white"
        )
        samples_seen = 0

        for images, labels in progress_bar:
            images = images.to(DEVICE)
            optimizer.zero_grad()
            reconstruction_logits, mu, log_var = model(images)

            loss, reconstruction_loss, kl_loss = vae_loss(images, reconstruction_logits, mu, log_var, beta=BETA)

            loss.backward()
            optimizer.step()

            samples_seen += images.size(0)
            total_loss += loss.item()
            total_kld_loss += kl_loss.item()
            total_reconstruction_loss += reconstruction_loss.item()

            progress_bar.set_postfix(
                loss=f"{total_loss / samples_seen:.3f}",
                recon=f"{total_reconstruction_loss / samples_seen:.3f}",
                kl=f"{total_kld_loss / samples_seen:.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        training_losses.append((total_loss / samples_seen, total_kld_loss / samples_seen, total_reconstruction_loss / samples_seen))
        validation_loss, validation_reconstruction, validation_kld = validate(model, validation_loader, BETA)
        validation_losses.append((validation_loss, validation_reconstruction, validation_kld))

        tqdm.write(
            f"Epoch {epoch + 1}: \n"
            f"validation loss = {validation_loss:.3f} \t"
            f"validation reconstruction = {validation_reconstruction:.3f} \t"
            f"validation kld = {validation_kld:.3f} \t"
        )

        scheduler.step()

        torch.save(model, 'weights.pth')

    return training_losses, validation_losses


def validate(model, validation_loader, BETA):
    model.eval()
    total_loss = 0
    total_reconstruction_loss = 0
    total_kld_loss = 0
    samples_seen = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(DEVICE)
            reconstruction_logits, mu, log_var = model(images, sample=False)

            loss, reconstruction_loss, kl_loss = vae_loss(images, reconstruction_logits, mu, log_var, beta=BETA)
            total_loss += loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_kld_loss += kl_loss.item()
            samples_seen += images.size(0)

    return (
        total_loss / samples_seen,
        total_reconstruction_loss / samples_seen,
        total_kld_loss / samples_seen
    )


def save_data(train_losses, validation_losses):
    to_save_df = {
        "training_losses": [],
        "training_reconstruction_losses": [],
        "training_kld_losses": [],
        "validation_losses": [],
        "validation_reconstruction_losses": [],
        "validation_kld_losses": [],
    }

    for train_loss, validation_loss in zip(train_losses, validation_losses):
        total_loss, reconstruction_loss, kl_loss = train_loss
        val_total_loss, val_reconstruction_loss, val_kl_loss = validation_loss
        to_save_df["training_losses"].append(total_loss)
        to_save_df["training_reconstruction_losses"].append(reconstruction_loss)
        to_save_df["training_kld_losses"].append(kl_loss)
        to_save_df["validation_losses"].append(val_total_loss)
        to_save_df["validation_reconstruction_losses"].append(val_reconstruction_loss)
        to_save_df["validation_kld_losses"].append(val_kl_loss)

    pd.DataFrame(to_save_df).to_csv("results.csv",sep=";", index=False)


if __name__ == "__main__":
    nn_config = load_config()
    train_set, val_set, test_set = dataloading(nn_config)
    training_loss, validation_loss = train(train_set, val_set, nn_config)
    # (training_loss, validation_loss)
    print(training_loss, validation_loss)
    save_data(training_loss, validation_loss)

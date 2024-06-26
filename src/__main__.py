from src.train.train_loop_h1 import train_vaegan
from src.data.ht1_data import get_dataset
from src.utils import config
from src.models.ht1_models import VAEGAN, Discriminator
import wandb

if __name__ == "__main__":

    wandb.init(project="vae-gan-triplet-mnist", config=config)

    train_dataloader, test_dataloader = get_dataset()

    vaegan = VAEGAN()
    discriminator = Discriminator()

    train_vaegan(config, vaegan, discriminator, train_dataloader, test_dataloader)

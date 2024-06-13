import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=5,
            padding=2,
            stride=2
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=64,
            momentum=0.9
        )
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            padding=2,
            stride=2
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=128,
            momentum=0.9
        )
        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=5,
            padding=2,
            stride=2
        )
        self.bn3 = nn.BatchNorm2d(
            num_features=256,
            momentum=0.9
        )
        self.relu = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear(
            in_features=256*8*8,
            out_features=2048
        )
        self.bn4 = nn.BatchNorm1d(
            num_features=2048,
            momentum=0.9
        )
        self.fc_mu = nn.Linear(
            in_features=2048,
            out_features=self.latent_dim
        )
        self.fc_logvar = nn.Linear(
            in_features=2048,
            out_features=self.latent_dim
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = x.view(batch_size, -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(
            in_features=self.latent_dim,
            out_features=8*8*256
        )
        self.bn1 = nn.BatchNorm1d(
            num_features=8*8*256,
            momentum=0.9
        )
        self.relu = nn.LeakyReLU(0.2)
        self.conv1 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=256,
            kernel_size=6,
            stride=2,
            padding=2
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=256,
            momentum=0.9
        )
        self.conv2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=6,
            stride=2,
            padding=2
        )
        self.bn3 = nn.BatchNorm2d(
            num_features=128,
            momentum=0.9
        )
        self.conv3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=32,
            kernel_size=6,
            stride=2,
            padding=2
        )
        self.bn4 = nn.BatchNorm2d(
            num_features=32,
            momentum=0.9
        )
        self.conv4 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = x.view(-1, 256, 8, 8)

        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            padding=2,
            stride=1
        )
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=128,
            kernel_size=5,
            padding=2,
            stride=2
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=128,
            momentum=0.9
        )
        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=5,
            padding=2,
            stride=2
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=256,
            momentum=0.9
        )
        self.conv4 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=5,
            padding=2,
            stride=2
        )
        self.bn3 = nn.BatchNorm2d(
            num_features=256,
            momentum=0.9
        )
        self.fc1 = nn.Linear(
            in_features=8*8*256,
            out_features=512
        )
        self.bn4 = nn.BatchNorm1d(
            num_features=512,
            momentum=0.9
        )
        self.fc2 = nn.Linear(
            in_features=512,
            out_features=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = x.view(-1, 256*8*8)
        x1 = x

        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.sigmoid(x)

        return x, x1


class VAEGAN(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAEGAN, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = Encoder(latent_dim=self.latent_dim)
        self.decoder = Decoder(latent_dim=self.latent_dim)
        self.discriminator = Discriminator()

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.discriminator.apply(weights_init)

    def forward(self, x):
        batch_size = x.shape[0]

        z_mu, z_logvar = self.encoder(x)
        std = z_logvar.mul(0.5).exp_()

        eps = Variable(torch.randn(batch_size, self.latent_dim)).to(device)

        z = z_mu + std * eps

        x_tilda = self.decoder(z)

        return z_mu, z_logvar, x_tilda

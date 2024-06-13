import torch
import torch.nn as nn
from torch.autograd import Variable


def train_vaegan(config, vaegan, discriminator, train_dataloader, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vaegan.to(device)
    discriminator.to(device)

    sample_train_batch = next(iter(train_dataloader))
    sample_test_batch = next(iter(test_dataloader))

    x_fixed = Variable(sample_train_batch[0]).to(device)
    z_fixed = Variable(torch.randn((config['batch_size'], config['latent_dim']))).to(device)

    encoder_optimizer = torch.optim.RMSprop(vaegan.encoder.parameters(), lr=config['learning_rate_encoder'])
    decoder_optimizer = torch.optim.RMSprop(vaegan.decoder.parameters(), lr=config['learning_rate_decoder'])
    discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=config['learning_rate_discriminator'])

    # encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.5, patience=5)
    # decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='min', factor=0.5, patience=5)
    # discriminator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(discriminator_optimizer, mode='min', factor=0.5, patience=5)

    bce = nn.BCELoss().to(device)

    for epoch in range(config['epochs']):
        prior_loss_list, gan_loss_list, recon_loss_list = [], [], []
        dis_real_list, dis_fake_list, dis_prior_list = [], [], []

        for i, (data, _) in enumerate(train_dataloader):

            ones_label_real = Variable(torch.ones(config['batch_size'], 1)).to(device)
            zeros_label_recon = Variable(torch.zeros(config['batch_size'], 1)).to(device)
            zeros_label_prior = Variable(torch.zeros(config['batch_size'], 1)).to(device)

            data = Variable(data).to(device)

            mu, logvar, recon = vaegan(data)

            z_p = Variable(torch.randn((config['batch_size'], config['latent_dim']))).to(device)

            x_p_tilda = vaegan.decoder(z_p)

            out = discriminator(data)[0]
            loss_d_real = bce(out, ones_label_real)
            dis_real_list.append(loss_d_real.item())

            out = discriminator(recon)[0]
            loss_d_rec = bce(out, zeros_label_recon)
            dis_fake_list.append(loss_d_rec.item())

            out = discriminator(x_p_tilda)[0]
            loss_d_prior = bce(out, zeros_label_prior)
            dis_prior_list.append(loss_d_prior.item())

            gan_loss = loss_d_real + loss_d_rec + loss_d_prior
            gan_loss_list.append(gan_loss.item())

            discriminator_optimizer.zero_grad()
            gan_loss.backward(retain_graph=True)
            discriminator_optimizer.step()

            out = discriminator(data)[0]
            loss_d_real = bce(out, ones_label_real)

            out = discriminator(recon)[0]
            loss_d_rec = bce(out, zeros_label_recon)

            out = discriminator(x_p_tilda)[0]
            loss_d_prior = bce(out, zeros_label_prior)
            gan_loss = loss_d_real + loss_d_rec + loss_d_prior

            x_l_tilda = discriminator(recon)[1]
            x_l = discriminator(data)[1]
            rec_loss = ((x_l_tilda - x_l) ** 2).mean()
            loss_dec = config['gamma'] * rec_loss - gan_loss
            recon_loss_list.append(rec_loss.item())
            decoder_optimizer.zero_grad()
            loss_dec.backward(retain_graph=True)
            decoder_optimizer.step()

            mu, logvar, recon = vaegan(data)
            x_l_tilda = discriminator(recon)[1]
            x_l = discriminator(data)[1]
            rec_loss = ((x_l_tilda - x_l) ** 2).mean()
            prior_loss = 1 + logvar - mu.pow(2) - logvar.exp()
            prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(mu.data)
            prior_loss_list.append(prior_loss.item())
            loss_enc = prior_loss + 5 * rec_loss

            encoder_optimizer.zero_grad()
            loss_enc.backward(retain_graph=True)
            encoder_optimizer.step()

            if i % 50 == 0:
                print(
                    '[%d/%d][%d/%d]\tLoss_gan: %.4f\tLoss_prior: %.4f\tRec_loss: %.4f\tdis_real_loss: %0.4f\tdis_fake_loss: %.4f\tdis_prior_loss: %.4f'
                    % (epoch, config['epochs'], i, len(train_dataloader),
                       gan_loss.item(), prior_loss.item(), rec_loss.item(), loss_d_real.item(), loss_d_rec.item(),
                       loss_d_prior.item()))

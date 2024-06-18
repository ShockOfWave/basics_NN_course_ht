import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import make_grid
import wandb


def train_vaegan(config, vaegan, discriminator, train_dataloader, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vaegan.to(device)
    discriminator.to(device)

    sample_train_batch = next(iter(train_dataloader))
    sample_test_batch = next(iter(test_dataloader))

    x_fixed = Variable(sample_train_batch[0]).to(device)
    z_fixed = Variable(torch.randn((config['batch_size'], config['latent_dim']))).to(device)

    x_val_fixed = Variable(sample_test_batch[0]).to(device)
    z_val_fixed = Variable(torch.randn((config['batch_size'], config['latent_dim']))).to(device)

    encoder_optimizer = torch.optim.RAdam(vaegan.encoder.parameters(), lr=config['learning_rate_encoder'])
    decoder_optimizer = torch.optim.RAdam(vaegan.decoder.parameters(), lr=config['learning_rate_decoder'])
    discriminator_optimizer = torch.optim.RAdam(discriminator.parameters(), lr=config['learning_rate_discriminator'])

    encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.5, patience=5)
    decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='min', factor=0.5, patience=5)
    discriminator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(discriminator_optimizer, mode='min', factor=0.5, patience=5)

    bce = nn.BCELoss().to(device)

    for epoch in range(config['epochs']):
        prior_loss_list, gan_loss_list, recon_loss_list = [], [], []
        dis_real_list, dis_fake_list, dis_prior_list = [], [], []

        loss_encoder, loss_decoder, loss_discriminator = [], [], []

        vaegan.train()
        discriminator.train()

        for i, (data, _) in enumerate(train_dataloader):

            ones_label_real = Variable(torch.ones(len(data), 1)).to(device)
            zeros_label_recon = Variable(torch.zeros(len(data), 1)).to(device)
            zeros_label_prior = Variable(torch.zeros(len(data), 1)).to(device)

            data = Variable(data).to(device)

            mu, logvar, recon = vaegan(data)

            z_p = Variable(torch.randn((len(data), config['latent_dim']))).to(device)

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
            loss_discriminator.append(gan_loss.item())

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
            loss_decoder.append(loss_dec.item())
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
            loss_encoder.append(loss_enc.item())

            encoder_optimizer.zero_grad()
            loss_enc.backward(retain_graph=True)
            encoder_optimizer.step()

            if i % 50 == 0:
                print(
                    'Train losses: [%d/%d][%d/%d]\tLoss_gan: %.4f\tLoss_prior: %.4f\tRec_loss: %.4f\tdis_real_loss: %0.4f\tdis_fake_loss: %.4f\tdis_prior_loss: %.4f'
                    % (epoch, config['epochs'], i, len(train_dataloader),
                       gan_loss.item(), prior_loss.item(), rec_loss.item(), loss_d_real.item(), loss_d_rec.item(),
                       loss_d_prior.item()))

        encoder_scheduler.step(sum(loss_encoder)/len(loss_encoder))
        decoder_scheduler.step(sum(loss_decoder)/len(loss_decoder))
        discriminator_scheduler.step(sum(loss_discriminator)/len(loss_discriminator))

        wandb.log({"epoch": epoch, "prior_loss": sum(prior_loss_list)/len(prior_loss_list)})
        wandb.log({"epoch": epoch, "gan_loss": sum(gan_loss_list) / len(gan_loss_list)})
        wandb.log({"epoch": epoch, "recon_loss": sum(recon_loss_list) / len(recon_loss_list)})
        wandb.log({"epoch": epoch, "dis_real": sum(dis_real_list) / len(dis_real_list)})
        wandb.log({"epoch": epoch, "dis_fake": sum(dis_fake_list) / len(dis_fake_list)})
        wandb.log({"epoch": epoch, "dis_prior": sum(dis_prior_list) / len(dis_prior_list)})
        wandb.log({
            'encoder_lr': encoder_optimizer.param_groups[0]['lr'],
            'decoder_lr': decoder_optimizer.param_groups[0]['lr'],
            'discriminator_lr': discriminator_optimizer.param_groups[0]['lr']
        })

        recon_fixed = vaegan(x_fixed)[2].detach()
        prior_fixed = vaegan.decoder(z_fixed).detach()

        wandb.log({"epoch": epoch,
                   "original_images": [wandb.Image(make_grid((sample_train_batch[0][:9] * 0.5 + 0.5).cpu(), 3))],
                   "recon_images": [wandb.Image(make_grid((recon_fixed * 0.5 + 0.5).cpu()[:9], 3))],
                   "generated_images": [wandb.Image(make_grid((prior_fixed * 0.5 + 0.5).cpu()[:9], 3))]})

        with torch.inference_mode():

            prior_loss_list, gan_loss_list, recon_loss_list = [], [], []
            dis_real_list, dis_fake_list, dis_prior_list = [], [], []

            loss_encoder, loss_decoder, loss_discriminator = [], [], []

            vaegan.eval()
            discriminator.eval()

            for i, (data, _) in enumerate(test_dataloader):
                ones_label_real = Variable(torch.ones(len(data), 1)).to(device)
                zeros_label_recon = Variable(torch.zeros(len(data), 1)).to(device)
                zeros_label_prior = Variable(torch.zeros(len(data), 1)).to(device)

                data = Variable(data).to(device)

                mu, logvar, recon = vaegan(data)

                z_p = Variable(torch.randn((len(data), config['latent_dim']))).to(device)

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
                loss_discriminator.append(gan_loss.item())

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
                loss_decoder.append(loss_dec.item())

                mu, logvar, recon = vaegan(data)
                x_l_tilda = discriminator(recon)[1]
                x_l = discriminator(data)[1]
                rec_loss = ((x_l_tilda - x_l) ** 2).mean()
                prior_loss = 1 + logvar - mu.pow(2) - logvar.exp()
                prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(mu.data)
                prior_loss_list.append(prior_loss.item())
                loss_enc = prior_loss + 5 * rec_loss
                loss_encoder.append(loss_enc.item())

                if i % 50 == 0 or i == len(test_dataloader):
                    print(
                        'Validation losses: [%d/%d][%d/%d]\tLoss_gan: %.4f\tLoss_prior: %.4f\tRec_loss: %.4f\tdis_real_loss: %0.4f\tdis_fake_loss: %.4f\tdis_prior_loss: %.4f'
                        % (epoch, config['epochs'], i, len(test_dataloader),
                           gan_loss.item(), prior_loss.item(), rec_loss.item(), loss_d_real.item(), loss_d_rec.item(),
                           loss_d_prior.item()))

            wandb.log({"epoch": epoch, "prior_val_loss": sum(prior_loss_list) / len(prior_loss_list)})
            wandb.log({"epoch": epoch, "gan_val_loss": sum(gan_loss_list) / len(gan_loss_list)})
            wandb.log({"epoch": epoch, "recon_val_loss": sum(recon_loss_list) / len(recon_loss_list)})
            wandb.log({"epoch": epoch, "dis_val_real": sum(dis_real_list) / len(dis_real_list)})
            wandb.log({"epoch": epoch, "dis_val_fake": sum(dis_fake_list) / len(dis_fake_list)})
            wandb.log({"epoch": epoch, "dis_val_prior": sum(dis_prior_list) / len(dis_prior_list)})

            recon_val_fixed = vaegan(x_val_fixed)[2].detach()
            prior_val_fixed = vaegan.decoder(z_val_fixed).detach()

            wandb.log({"epoch": epoch,
                       "original_val_images": [wandb.Image(make_grid((sample_test_batch[0][:9] * 0.5 + 0.5).cpu(), 3))],
                       "recon_val_images": [wandb.Image(make_grid((recon_val_fixed * 0.5 + 0.5).cpu()[:9], 3))],
                       "generated_val_images": [wandb.Image(make_grid((prior_val_fixed * 0.5 + 0.5).cpu()[:9], 3))]})

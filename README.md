# Basics neural network course home tasks

### Project structure

```
basics_NN_course_ht/
├── data/                       <- Project data
├── src/                        <- Project source code
│ ├── data/                     <- Source code for data processing
│ ├── models/                   <- Source code for models
│ ├── train/                    <- Source code for train loops
│ ├── __main__.py               <- Init train file
│ ├── logs_to_gif.py            <- Wandb logs to gif files
│ └── utils.py                  <- Config and paths
├── images/                     <- gifs and images for readme
├── train_rnn.py                <- script for RNN, GRU and LSTM training
├── Steel_industry_data.csv     <- data for RNN, GRU and LSTM training
├── README.md
└── requirements.txt
```

### First task: VAE+GAN

##### Develop a bundle of VAE and GAN to recover numbers from the MNIST dataset. In this case, VAE and GAN are not separate neural networks, but form a single pipeline with a triplex loss function

[//]: # (<p align="center">)

[//]: # (    <figure style="display: inline-block; margin: 0 10px; text-align: center">)

[//]: # (        <img src="images/real_images.gif" alt="real images" width="200"/>)

[//]: # (        <figcaption style="font-weight: bold; font-size: 16px">Real images</figcaption>)

[//]: # (    </figure>)

[//]: # (    <figure style="display: inline-block; margin: 0 10px; text-align: center">)

[//]: # (        <img src="images/recon_images.gif" alt="real images" width="200"/>)

[//]: # (        <figcaption style="font-weight: bold; font-size: 16px">Reconstructed images</figcaption>)

[//]: # (    </figure>)

[//]: # (    <figure style="display: inline-block; margin: 0 10px; text-align: center">)

[//]: # (        <img src="images/generated_images.gif" alt="real images" width="200"/>)

[//]: # (        <figcaption style="font-weight: bold; font-size: 16px">Generated images</figcaption>)

[//]: # (    </figure>)

[//]: # (</p>)


<table>
  <tr>
    <td align="center">
      <img src="images/real_images.gif" alt="real_images" width="200"/><br>
      <b style="font-size: 14px;">Original images</b>
    </td>
    <td align="center">
      <img src="images/recon_images.gif" alt="recon_images" width="200"/><br>
      <b style="font-size: 14px;">Recon images</b>
    </td>
    <td align="center">
      <img src="images/generated_images.gif" alt="generated_images" width="200"/><br>
      <b style="font-size: 14px;">Generated images</b>
    </td>
  </tr>
</table>

### Second task: RNN

Develop RNN, GRU and LSTM to predict Usage_kWh. Dataset - http://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption.

Hyperparameters are at your discretion

Compare the quality of the MSE, RMSE and R^2 models

### Acknowledgement and literature

#### VAE+GAN:
```
@ARTICLE{Larsen2015-yc,
  title         = "Autoencoding beyond pixels using a learned similarity metric",
  author        = "Larsen, Anders Boesen Lindbo and S{\o}nderby, S{\o}ren Kaae
                   and Larochelle, Hugo and Winther, Ole",
  month         =  dec,
  year          =  2015,
  copyright     = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
  archivePrefix = "arXiv",
  primaryClass  = "cs.LG",
  eprint        = "1512.09300"
}
   ```

https://github.com/rishabhd786/VAE-GAN-PYTORCH

#### RNN

```
@ARTICLE{Schmidt2019-ai,
  title         = "Recurrent Neural Networks ({RNNs)}: A gentle introduction
                   and overview",
  author        = "Schmidt, Robin M",
  month         =  nov,
  year          =  2019,
  copyright     = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
  archivePrefix = "arXiv",
  primaryClass  = "cs.LG",
  eprint        = "1912.05911"
}

```

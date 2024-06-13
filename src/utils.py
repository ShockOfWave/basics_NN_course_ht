from pathlib import Path
import torch


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def add_units_to_conv2d(x, y, unique_values=10):
    dim1 = int(x.shape[2])
    dim2 = int(x.shape[3])
    dimc = unique_values

    repeat_n = dim1*dim2
    main_tmp = torch.zeros((x.shape[0], 360))

    for i, value in enumerate(y):
        tmp_list = torch.zeros(10)
        tmp_list[value] = 1
        tmp_list = tmp_list.repeat(repeat_n)
        main_tmp[i] = tmp_list

    y_repeat = torch.reshape(main_tmp, (x.shape[0], dimc, dim1, dim2))
    return torch.cat([x, y_repeat], 1)


config = {
        'epochs': 50,
        'learning_rate_encoder': 3e-4,
        'learning_rate_decoder': 3e-4,
        'learning_rate_discriminator': 3e-5,
        'alpha': 0.1,
        'gamma': 15,
        'batch_size': 256,
        'latent_dim': 128
    }
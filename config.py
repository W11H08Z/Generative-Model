import argparse

def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--model", type=str, help="The name of model")
    parser.add_argument("--save_model_path", type=str)
    parser.add_argument("--dataset_path", type=str)

    return parser.parse_args()
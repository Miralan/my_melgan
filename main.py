import argparse
from solve_melgan import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default='wavs')
    parser.add_argument("--save_path", type=str, default='save')
    parser.add_argument("--n_mel", type=int, default=80)
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)

    parser.add_argument("--ndf", type=int, default=16)
    parser.add_argument("--num_D", type=int, default=3)
    parser.add_argument("--n_layers_D", type=int, default=4)
    parser.add_argument("--downsamp_factor", type=int, default=4)
    parser.add_argument("--lambda_feat", type=float, default=10)

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=8192)

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--n_test_samples", type=int, default=8)

    config = parser.parse_args()

    train(config)


if __name__ == '__main__':
    main()

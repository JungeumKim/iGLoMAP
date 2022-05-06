from __future__ import print_function
import argparse

from iglo._data import shortest_path_savior


# worktree: enc_44d9b36

def parse_args():
    # Arguments:
    parser = argparse.ArgumentParser(description='Arguments.')

    # new arguments:
    parser.add_argument('--dataset', default="mnist_np",
                        help='["mnist_np", ...]')
    parser.add_argument('--N', type=int, default=6000)
    parser.add_argument('--PP', type=int, default=15)
    args = parser.parse_args()
    return args


def main(args):
    path_D, path_d, path_press = shortest_path_savior(args.dataset,
                         args.N,
                         max_=True,
                         n_neighbors=args.PP)
    print("shortest saving done.")

if __name__ == '__main__':
    args = parse_args()

    main(args)
    exit()



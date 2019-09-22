import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoches", type=int, default=300, help="number of epochs")
parser.add_argument("--lr", type=int, default=3e-4, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=3, help="size of each image batch")
parser.add_argument("--frames", type=int, default=64, help="size of cut continuious frames")
parser.add_argument("--cut_height", type=int, default=64, help="size of cut height")
parser.add_argument("--cut_width", type=int, default=64, help="size of cut width")
parser.add_argument("--evaluation_interval", type=int, default=10, help="interval evaluations on validation set")
parser.add_argument("--contour_reg_flag", type=int, default=1, help="if use contour regression")
opt = parser.parse_args()

if __name__ == '__main__':
    print(opt.epochs)
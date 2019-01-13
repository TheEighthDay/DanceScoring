import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch DanceScore')

"----------------------------- General options -----------------------------"
parser.add_argument('--env', default='test_score', type=str,help='the environment of the visdom')
parser.add_argument('--model_path', default='weight/resnet10_2.pth', type=str,help='the .pth file save position')
parser.add_argument('--train', default=False, type=bool,help='train or test')
parser.add_argument('--image_size', default=224, type=int,help='image size')
parser.add_argument('--workers', default=0, type=int,help='workers multiprocess')
parser.add_argument('--no_cuda', default=False, type=bool,help='If true, cuda is not used.')
parser.add_argument('--train_triplet_txt', default='data/txt/train_2.txt', type=str,help='train_triplet_txt')
parser.add_argument('--validation_triplet_txt', default='data/txt/validation_2.txt', type=str,help='validation_triplet_txt')
parser.add_argument('--test_triplet_txt', default='data/txt/test_2.txt', type=str,help='test_triplet_txt')





"----------------------------- Model options -----------------------------"
parser.add_argument('--mode', default='feature',type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
parser.add_argument('--model_depth', default=10, type=int, help='Depth of resnext (50 | 101| 152)')
parser.add_argument('--model_name', default='resnet', type=str,help='the model')
parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
parser.add_argument('--n_classes', default=101, type=int, help='classfity number')
parser.add_argument('--margin', default=10, type=float, help='the margin of triplet')
parser.add_argument('--p', default=2, type=int, help='the norm degree of triplet')
parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
parser.add_argument('--lr', default=0.000004, type=float, help='learning rate')
parser.add_argument('--lr_decay', default=5, type=int, help='decay coefficient of learning rate ')
parser.add_argument('--epochs', default=30, type=int, help='epochs')
parser.add_argument('--print_freq', default=5, type=int, help='print_freq')
parser.add_argument('--lowest_loss', default=1000, type=int, help='lowest_loss')
parser.add_argument('--sample_size', default=112, type=int, help='sample_size')
parser.add_argument('--sample_duration', default=16, type=int, help='sample_duration')






opt = parser.parse_args()


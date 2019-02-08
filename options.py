import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PlaneFlow')

    parser.add_argument('--data_path', dest='data_path',
                        help='path to data',
                        default='.', type=str)
    parser.add_argument('--corner_type', dest='corner_type',
                        help='corner type for the search algorithm',
                        default='annots_only', type=str)
    parser.add_argument('--testing_corner_type', dest='testing_corner_type',
                        help='testing corner type for the search algorithm',
                        default='dets_only', type=str)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='batch size',
                        default=16, type=int)
    parser.add_argument('--num_edges', dest='num_edges',
                        help='num edges',
                        default=0, type=int)
    parser.add_argument('--num_training_images', dest='num_training_images',
                        help='num training images',
                        default=10000, type=int)
    parser.add_argument('--num_testing_images', dest='num_testing_images',
                        help='num testing images',
                        default=-1, type=int)
    parser.add_argument('--restore', dest='restore',
                        help='restore',
                        default=1, type=int)
    parser.add_argument('--LR', dest='LR',
                        help='learning rate',
                        default=1e-3, type=float)
    parser.add_argument('--task', dest='task',
                        help='task',
                        default='train', type=str)
    parser.add_argument('--max_num_edges', dest='max_num_edges',
                        help='maximum number of_edges',
                        default=20, type=int)
    parser.add_argument('--suffix', dest='suffix',
                        help='the suffix to distinguish experiments with different configurations, ["", "mixed", "uniform", "single"]',
                        default='', type=str)
    parser.add_argument('--conv_type', dest='conv_type',
                        help='convolution type for gnn',
                        default='', type=str)

    return parser.parse_args()

import os
import argparse
from torch.backends import cudnn
import torch
from data.loader import get_train_loader, get_test_list
from solver import Solver

#jayeon

def main(args):
    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:
        fig_path = os.path.join(args.save_path, 'fig')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))

    # data_loader = get_loader(mode=args.mode,
    #                          load_mode=args.load_mode,
    #                          saved_path=args.saved_path,
    #                          test_patient=args.test_patient,
    #                          patch_n=(args.patch_n if args.mode=='train' else None),
    #                          patch_size=(args.patch_size if args.mode=='train' else None),
    #                          transform=args.transform,
    #                          batch_size=(args.batch_size if args.mode=='train' else 1),
    #                          num_workers=args.num_workers)
    train_data_loader = get_train_loader(args, mode = args.mode)
    test_img_list = get_test_list(args,mode = args.mode)

    solver = Solver(args, train_data_loader, test_img_list)
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()


if __name__ == "__main__":

    data_dir = r'../../data/denoising'
    checkpoint_dir = os.path.join(data_dir, 'checkpoint')
    test_result_dir = os.path.join(data_dir, 'test-results')
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help="train | test")
    parser.add_argument('--load_mode', type=int, default=0, help="0 | 1")

    parser.add_argument('--data_path', type=str, default='./AAPM-Mayo-CT-Challenge/')
    parser.add_argument('--saved_path', type=str, default='./npy_img/')
    parser.add_argument('--save_path', type=str, default='./save/')
    parser.add_argument('--test_patient', type=str, default='L506')
    parser.add_argument('--result_fig', type=bool, default=True)

    # parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    # parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--norm_range_min', type=float, default=-1000.0)
    parser.add_argument('--norm_range_max', type=float, default=400.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)

    parser.add_argument('--transform', type=bool, default=False)
    # if patch training, batch size is (--patch_n x --batch_size)
    parser.add_argument('--patch_n', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--decay_iters', type=int, default=3000)
    parser.add_argument('--save_iters', type=int, default=1000)
    parser.add_argument('--test_iters', type=int, default=1000)

    parser.add_argument('--n_d_train', type=int, default=4)

    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--lambda_', type=float, default=10.0)

    parser.add_argument('--device', type=str, default = 'cuda')
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--multi_gpu', type=bool, default=True)

    parser.add_argument('--dataset', type = str, default = 'mayo')
    parser.add_argument('--swt', type = bool, default = False)
    parser.add_argument('--n_channels', type = int, default = 1)
    parser.add_argument('--benchmark', type = bool, default = False)
    parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
    parser.add_argument('--thickness', type=int, default=3,
                    help='Specify thicknesses of mayo dataset (1 or 3 mm)')
    parser.add_argument('--data_dir', type=str, default=data_dir,
                    help='Path of training directory contains both lr and hr images')
    parser.add_argument('--in_mem', default=False, action='store_true',
                    help="Load whole data into memory, Default: False")
    parser.add_argument('--ext', type=str, default='sep',
                    help='File extensions')
    parser.add_argument('--train_datasets', type = str, default = 'mayo')
    parser.add_argument('--augment', type=bool, default = True,
                    help='Do random flip (vertical, horizontal, rotation)')
    parser.add_argument('--model', type = str, default = 'wganvgg')
    parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir,
                    help='Path to checkpoint directory')
    parser.add_argument('--resume',default=False, action='store_true' )
    parser.add_argument('--resume_best',default=False, action='store_true' )
    parser.add_argument('--path_opt', default='', type=str,
                    help="Specify options in path name")
    parser.add_argument('--log_file', default='', type = str)
    parser.add_argument('--test_result_dir', default=test_result_dir, type =str)
    parser.add_argument('--patch_offset', type=int, default=5,
                    help='Size of patch offset')

    args = parser.parse_args()
    torch.manual_seed(123)
    print(args)
    main(args)

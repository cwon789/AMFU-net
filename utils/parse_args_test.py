from utils.utils import *

def parse_args():
    """Testing Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='AMFU-net for IRSTD')

    # choose model
    parser.add_argument('--model', type=str, default='AMFU',
                        help='AMFU, AMFU_dilation, AMFU_noATN, AMFU_noResATN')
    
    # Deep supervision for AMFU-nets              
    parser.add_argument('--deep_supervision', type=str, default='DSV', help='DSV or None')               ######## ACM --> None

    # data and pre-process
    parser.add_argument('--dataset', type=str, default='NUAA-SIRST',
                        help='dataset name: NUDT-SIRST, NUAA-SIRST, NUST-SIRST')
    parser.add_argument('--st_model', type=str, default='AMFU',                             ######## Change
                        help='NUDT-SIRST_DNANet_31_07_2021_14_50_57_wDS,'
                             'NUAA-SIRST_DNANet_28_07_2021_05_21_33_wDS')                         
    parser.add_argument('--model_dir', type=str,                                                         ######## Change
                        default = './AMFU-net/result/NUAA-SIRST_AMFU/AMFU_epoch.pth.tar',      # Trained weight directory
                        help    = 'NUDT-SIRST_DNANet_31_07_2021_14_50_57_wDS/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar,'
                                  'NUAA-SIRST_DNANet_28_07_2021_05_21_33_wDS/mIoU__DNANet_NUAA-SIRST_epoch.pth.tar')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when --mode==Ratio')
    parser.add_argument('--root', type=str, default='./AMFU-net/dataset')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split_method', type=str, default='50_50',
                        help='50_50, 10000_100(for NUST-SIRST)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop image size')

    #  hyper params for training
    parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                        help='number of epochs to train (default: 110)')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')

    # select GPUs
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')

    # ROC threshold number of image
    parser.add_argument('--ROC_thr', type=int, default=10,
                        help='crop image size')

    # the parser
    args = parser.parse_args()

    return args
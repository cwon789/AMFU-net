# Basic module
import cv2

# Torch and visulization
from torchvision      import transforms

# Metric, loss .etc
from utils.utils import *
from utils.loss import *
from utils.load_param_data import load_param

# my model
from models.AMFU import *
from models.AMFU_noATN import *
from models.AMFU_noResATN import *


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')
    # choose model
    parser.add_argument('--model', type=str, default='AMFU',                                  ######## Change
                        help='AMFU, AMFU_noATN, AMFU_noResATN')
    parser.add_argument('--deep_supervision', type=str, default='DSV', help='DSV or None')        ######## Change

    # data and pre-process
    parser.add_argument('--img_demo_dir', type=str, default='/your/own/path',
                        help='img_demo')
    parser.add_argument('--img_demo_index', type=str,default='Misc_212',
                        help='target1, target2, target3')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when --mode==Ratio')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop image size')

    #  hyper params for training
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')

    # cuda and logging
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')

    args = parser.parse_args()

    # the parser
    return args

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        img_dir   = args.img_demo_dir+'/'+args.img_demo_index+args.suffix

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        data            = DemoLoader (img_dir, base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        img             = data.img_preprocess()
        img_origin      = args.img_demo_dir+'/'+args.img_demo_index+args.suffix

        # Network selection
        if args.model   == 'AMFU':
            model       = AMFU()
        
        elif args.model == 'AMFU_noATN':                      
            model       = AMFU_noATN()

        elif args.model == 'AMFU_noResATN':
            model       = AMFU_noResATN()

        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Load Checkpoint
        checkpoint      = torch.load('/home/jay/catkin_GRSL/3.Detection_compare/DNA_net/result_before/NUAA-SIRST_UNet3_CGM/mIoU__UNet3_CGM_NUAA-SIRST_epoch.pth.tar')
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        img = img.cuda()
        img = torch.unsqueeze(img,0)

        # For resizing
        read_origin = cv2.imread(img_origin)
        W,H,_ = read_origin.shape

        if args.deep_supervision == 'DSV':
            preds = self.model(img)
            pred  = preds[-1]
        else:
            pred  = self.model(img)

        # save_Pred_GT_visulize(pred, args.img_demo_dir, args.img_demo_index, args.suffix, H, W)
        save_Pred_GT_IoU_visulize(pred, args.img_demo_dir, args.img_demo_index, args.suffix, H, W, 0.3)
        # save_Pred_GT_visulize2(pred, args.img_demo_dir, args.img_demo_index, args.suffix, H, W)


def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    print('---------------------',args.model, '---------------------')
    main(args)






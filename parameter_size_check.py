from tqdm             import tqdm
from utils.parse_args_test import *
import scipy.io as scio

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from utils.utils import *
from utils.metric import *
from utils.loss import *
from utils.load_param_data import  load_dataset, load_param

# my model
from models.AMFU import *
from models.AMFU_noATN import *
from models.AMFU_noResATN import *

import torchsummary

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, args.ROC_thr)
        self.PD_FA = PD_FA(1,args.ROC_thr)
        self.mIoU  = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

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
        torchsummary.summary(model, (3, 256, 256),device='cuda')

def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    print('---------------------',args.model, '---------------------')
    main(args)

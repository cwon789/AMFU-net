# import torch and utils
from tqdm             import tqdm
import torch.optim    as optim
from torchvision      import transforms
from torch.utils.data import DataLoader
from utils.parse_args_train import  parse_args

# metric, loss .etc
from utils.utils import *
from utils.metric import *
from utils.loss2 import *
from utils.load_param_data import  load_dataset

# my model
from models.AMFU import *
from models.AMFU_noATN import *
from models.AMFU_noResATN import *



class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.ROC  = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        trainset        = TrainSetLoader(dataset_dir,img_id=train_img_ids,base_size=args.base_size,crop_size=args.crop_size,transform=input_transform,suffix=args.suffix)
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Network selection
        # my network
        if args.model   == 'AMFU':
            model       = AMFU()
        
        elif args.model == 'AMFU_noATN':                      
            model       = AMFU_noATN()

        elif args.model == 'AMFU_noResATN':
            model       = AMFU_noResATN()
        
        # model init
        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Optimizer and lr scheduling
        if args.optimizer   == 'Adam':
            self.optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer  = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
           
        # Evaluation metrics
        self.best_iou       = 0
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

    # Training
    def training(self,epoch):

        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()

        for i, ( data, labels) in enumerate(tbar):
            data   = data.cuda()
            labels = labels.cuda()

            if args.deep_supervision == 'DSV':
                preds= self.model(data)
                loss = 0
                for pred in preds:
                    loss += SoftIoULoss(pred, labels)
                    # loss += FocalIoULoss(pred, labels)
                loss /= len(preds)

            elif args.deep_supervision == 'None':
               pred = self.model(data)
               loss = SoftIoULoss(pred, labels)
            #    loss = FocalIoULoss(pred, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))

        self.train_loss = losses.avg

    # Testing
    def testing (self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        losses = AverageMeter()

        with torch.no_grad():
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()

                if args.deep_supervision == 'DSV':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                        # loss += FocalIoULoss(pred, labels)
                    loss /= len(preds)
                    pred =preds[-1]

                elif args.deep_supervision == 'None':
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)
                    # loss = FocalIoULoss(pred, labels)
                    
                losses.update(loss.item(), pred.size(0))
                self.ROC .update(pred, labels)
                self.mIoU.update(pred, labels)
                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                _, mean_IOU = self.mIoU.get()
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU ))
                
            test_loss=losses.avg
        
        # save high-performance model
        save_model(mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                   self.train_loss, test_loss, recall, precision, epoch, self.model.state_dict())

def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)


if __name__ == "__main__":
    args = parse_args()
    print('---------------------',args.model,'---------------------')
    main(args)
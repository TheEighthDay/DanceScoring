from opt import opt
from dataset import PoseTripletValidation,PoseTripletTrain,PoseTripletTest,PoseTripletPredict
from visualize import Visualizer
from model import generate_model
from utils.net_utils import AverageMeter
from torch.utils.data import DataLoader
from epoch_op import train_epoch,validation_epoch,test_epoch
import torch.nn as nn
import torch.optim as optim
import torch


torch.manual_seed(666)
torch.cuda.manual_seed_all(666)

def main():
    vis=Visualizer(opt.env)
    if opt.train:
        vis.log("select train\n")

        vis.log("loading model\n")
        model=generate_model(opt)
        # print(model.named_parameters)
        if opt.model_path!='':
            model_data=torch.load(opt.model_path)
            model_data=model_data['state_dict']
            # del model_data['module.fc.weight']
            # del model_data['module.fc.bias']
            # print([k for k,v in model_data['state_dict'].items()])
            model.load_state_dict(model_data)
            # for param in list(model.parameters())[:-2]:
            #     param.requires_grad = False
            # for param in model.parameters():
            #     param.requires_grad = False
            # for param in model.module.fc.parameters():
            #     param.requires_grad = True
            # channel_in = model.module.fc.in_features
            # model.module.fc= nn.Linear(channel_in,1000)

            # for param in model.fcc.parameters():
            #     param.requires_grad = True

            vis.log("load model data from {}\n".format(opt.model_path))
        vis.log("load model over\n")


        vis.log("generate  data  iterator\n")
        train_loader = DataLoader(PoseTripletTrain(), batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=opt.workers)
        validation_loader = DataLoader(PoseTripletValidation(), batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=opt.workers)
        vis.log("generate  data  iterator  over\n")


        vis.log("generate triplet_loss function\n")
        if not opt.no_cuda:
            triplet_loss = nn.TripletMarginLoss(margin=opt.margin, p=opt.p).cuda()
        else:
            triplet_loss = nn.TripletMarginLoss(margin=opt.margin, p=opt.p)

        vis.log("generate triplet_loss function over\n")

        vis.log("generate optimizer\n")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr, weight_decay=opt.lr_decay, amsgrad=True)
        vis.log("generate optimizer over\n")

        


        vis.log("begin train...\n")
        for epoch in range(opt.epochs):
            train_epoch(train_loader,model,triplet_loss,optimizer,epoch,vis)
            low_loss=validation_epoch(validation_loader,model,triplet_loss,epoch,vis)
            vis.plot("low_loss",low_loss)

            
            state={
            'epoch':epoch,
            'state_dict': model.state_dict(),
            'lowest_loss': low_loss,
            'lr': opt.lr,
            }
            torch.save(state,"weight/"+opt.model_name+str(opt.model_depth)+"_"+str(epoch)+".pth")
            vis.log("model has been saved")
            

            if((epoch-1)>0 and epoch%5==0):
                opt.lr = opt.lr / opt.lr_decay
                optimizer = optim.Adam(model.parameters(), opt.lr, weight_decay=opt.weight_decay, amsgrad=True)
    else:
        vis.log("select test\n")


        vis.log("loading model\n")
        model=generate_model(opt)
        model_data=torch.load(opt.model_path)
        model.load_state_dict(model_data['state_dict'])
        vis.log("load model data from {}\n".format(opt.model_path))
        vis.log("load model over\n")

        vis.log("generate  data  iterator\n")
        test_loader = DataLoader(PoseTripletTest(), batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=opt.workers) #PoseTripletPredict
        vis.log("generate  data  iterator  over\n")


        vis.log("generate triplet_loss function\n")
        if not opt.no_cuda:
            triplet_loss = nn.TripletMarginLoss(margin=opt.margin, p=opt.p).cuda()
        else:
            triplet_loss = nn.TripletMarginLoss(margin=opt.margin, p=opt.p)

        vis.log("generate triplet_loss function over\n")


        test_epoch(test_loader,model,triplet_loss,vis)
        
       
if __name__ == '__main__':
    main()





















        

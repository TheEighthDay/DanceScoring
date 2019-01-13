from utils.net_utils import AverageMeter
from opt import opt

from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
def train_epoch(train_loader, model, criterion, optimizer, epoch,vis):
    vis.log("the {} epoch begin train..\n".format(epoch))

    vis.log("change model to train mode\n")
    model.train()
    vis.log("change model to train mode over\n")

    losses = AverageMeter()
    distpes = AverageMeter()
    distnes = AverageMeter()
   


    for i, (anchor,positive,negative) in tqdm(enumerate(train_loader)):
        if not opt.no_cuda:
            anchor=anchor.cuda()
            positive=positive.cuda()
            negative=negative.cuda()

        # anchor=Variable(anchor,requires_grad=True)
        # positive=Variable(positive,requires_grad=True)
        # negative=Variable(negative,requires_grad=True)
        batch_size=anchor.size()[0]

        anchor_f=model(anchor)
        positive_f=model(positive)
        negative_f=model(negative)

        loss=criterion(anchor_f,positive_f,negative_f)
        losses.update(loss.data[0],batch_size)

        distp=F.pairwise_distance(anchor_f,positive_f)
        distpes.update(distp.sum().data[0],batch_size)

        distn=F.pairwise_distance(anchor_f,negative_f)
        distnes.update(distn.sum().data[0],batch_size)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % opt.print_freq == 0:
            vis.plot("train_loss", losses.avg)
            vis.plot("train_dist_a_p", distpes.avg)
            vis.plot("train_dist_a_n", distnes.avg)
            vis.plot("train_distp/distn", distpes.avg/distnes.avg)


    vis.log("the {} epoch train over\n".format(epoch))





def validation_epoch(validation_loader, model, criterion, epoch,vis):
    vis.log("the {} epoch begin validation..\n".format(epoch))

    vis.log("change model to eval mode\n")
    model.eval()
    vis.log("change model to eval mode over\n")

    losses = AverageMeter()
    distpes = AverageMeter()
    distnes = AverageMeter()

    


    for i, (anchor,positive,negative) in tqdm(enumerate(validation_loader)):
        if not opt.no_cuda:
            anchor=anchor.cuda()
            positive=positive.cuda()
            negative=negative.cuda()

        # anchor=Variable(anchor,requires_grad=True)
        # positive=Variable(positive,requires_grad=True)
        # negative=Variable(negative,requires_grad=True)
        batch_size=anchor.size()[0]

        anchor_f=model(anchor)
        positive_f=model(positive)
        negative_f=model(negative)

        loss=criterion(anchor_f,positive_f,negative_f)
        losses.update(loss.data[0],batch_size)

        distp=F.pairwise_distance(anchor_f,positive_f)
        distpes.update(distp.sum().data[0],batch_size)

        distn=F.pairwise_distance(anchor_f,negative_f)
        distnes.update(distn.sum().data[0],batch_size)

        
        vis.plot("eval_loss", losses.avg)
        vis.plot("eval_dist_dist_a_p", distpes.avg)
        vis.plot("eval_dist_dist_a_n", distnes.avg)
        vis.plot("eval_distp/distn", distpes.avg/distnes.avg)


    vis.log("the {} epoch eval over\n".format(epoch))

    return losses.avg


def test_epoch(test_loader, model, criterion,vis):
    vis.log(" begin test..\n")

    vis.log("change model to eval mode\n")
    model.eval()
    vis.log("change model to eval mode over\n")


    losses = AverageMeter()
    distpes = AverageMeter()
    distnes = AverageMeter()
    accuracy = AverageMeter()
    score_p = AverageMeter()
    score_n = AverageMeter()

    

    for i, (anchor,positive,negative) in tqdm(enumerate(test_loader)):
        if not opt.no_cuda:
            anchor=anchor.cuda()
            positive=positive.cuda()
            negative=negative.cuda()

        # anchor=Variable(anchor,requires_grad=True)
        # positive=Variable(positive,requires_grad=True)
        # negative=Variable(negative,requires_grad=True)
        batch_size=anchor.size()[0]

        anchor_f=model(anchor)
        positive_f=model(positive)
        negative_f=model(negative)

        loss=criterion(anchor_f,positive_f,negative_f)
        losses.update(loss.data[0],batch_size)

        distp=F.pairwise_distance(anchor_f,positive_f)
        distpes.update(distp.sum().data[0],batch_size)

        distn=F.pairwise_distance(anchor_f,negative_f)
        distnes.update(distn.sum().data[0],batch_size)


        
        vis.plot("test_loss_avg", losses.avg)
        vis.plot("test_dist_a_p_avg", distpes.avg)
        vis.plot("test_dist_a_n_avg", distnes.avg)
        vis.plot("test_distp/distn_avg", distpes.avg/distnes.avg)

        if (distpes.val<distnes.val):
            accuracy.update(1,batch_size)
        else:
            accuracy.update(0,batch_size)


        def get_real_score(val):  #利用3次方函数扩大距离
            if val>55:
                val=55
            if val<31:
                val=31
            return (2-((val-31)/12))**3*12.5

        
        score_p.update(get_real_score(distpes.val),batch_size)
        score_n.update(get_real_score(distnes.val),batch_size)

   


        vis.plot("accuracy", accuracy.avg)
        vis.plot("score_p", score_p.avg)
        vis.plot("score_n", score_n.avg)


    vis.log("test over\n")



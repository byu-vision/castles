'''Model definition and LightningModules for classification and image
retrieval experiments'''
from argparse import ArgumentParser
import math

import torch, torchvision
import pytorch_lightning as pl
from PIL import Image

import dataset
    
################################################
# MODEL
################################################

class Classifier(torch.nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm1d(inc)
        self.drop1 = torch.nn.Dropout(0.25)
        self.fc1 = torch.nn.Linear(inc, inc//2)
        self.bn2 = torch.nn.BatchNorm1d(inc//2)
        self.drop2 = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(inc//2, outc)
    
    def forward(self, x):
        relu = torch.nn.functional.relu_
        x = relu(self.fc1(self.drop1(self.bn1(x))))
        x = self.fc2(self.drop2(self.bn2(x)))
        return x
    

def resnet(nclass, net='resnet50', pretrained=True, classifier=True):
    net = getattr(torchvision.models, net)(pretrained=pretrained)
    if classifier:
        net.fc = Classifier(net.fc.in_features, nclass)
    else:
        net.fc = torch.nn.Linear(net.fc.in_features, nclass)
    return net


################################################
# METRICS
################################################

@torch.no_grad()
def calc_accuracy(x, y):
    '''Classifcation accuracy'''
    return x.argmax(1).eq(y).float().mean()


@torch.no_grad()
def map_r(x, y, npos):
    '''Mean Average Precision at R'''
    r = npos-1
    n = x.shape[0]
    
    dmat = torch.cdist(x,x)
    rank = dmat.argsort(1)
    ymat = y[rank]
    match = ymat.eq(y.view(-1,1)).float()[:,1:] # remove first column (self)
    apr = match[:,:r].cumsum(1)
    apr *= match[:,:r]
    apr *= (1 / torch.arange(1, apr.shape[1]+1).to(apr))
    apr = apr.sum(1).div(r)
    return apr.mean()

################################################
# LIGHTNING
################################################

class CastleClassifier(pl.LightningModule):
    '''LightningModule for castle instance recognition/classification'''
    def __init__(self, nclass, hp, **kwargs):
        super().__init__()
        self.hp = hp
        self.hparams = hp
        self.nclass = nclass
        self._set_net(hp)
        self._set_lossfn(hp)
        
    def _set_net(self, hp):
        self.net = resnet(self.nclass, hp.net)
        
    def _set_lossfn(self, hp):
        self.lossfn = getattr(torch.nn, hp.loss)()
        
    def forward(self, x):
        return self.net(x)
        
    @staticmethod
    def add_model_specific_args(parent):
        parser = ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--data', type=str, default=dataset.DATA_ROOT)
        parser.add_argument('--net', type=str, default='resnet50')
        parser.add_argument('--optim', type=str, default='Adam')
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--bs', type=int, default=32)
        parser.add_argument('--workers', type=int, default=4)
        parser.add_argument('--loss', type=str, default='CrossEntropyLoss')
        
        return parser
        
    def configure_optimizers(self):
        opt = getattr(torch.optim, self.hp.optim)
        return opt(self.parameters(), lr=self.hp.lr)
    
    def optimizer_step(self, current_epoch, batch_nb, optimizer,
                       optimizer_idx, second_order_closure=None, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        # linear warmup to cosine decay
        total = self.trainer.max_epochs * len(self.trainer.train_dataloader)
        cut = int(0.3 * total)
        t = self.trainer.global_step
        lm = self.hp.lr
        if t < cut:
            lr_scale = min(1., float(t + 1) / cut)
            lr = lr_scale * lm
        else:
            lr = 0.5*self.hp.lr*(1+math.cos(math.pi*(t-cut)/(total-cut)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.step()
        optimizer.zero_grad()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.net(x)
        loss = self.lossfn(logits, y)
        acc = calc_accuracy(logits, y)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        result.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return result
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.net(x)
        loss = self.lossfn(logits, y)
        acc = calc_accuracy(logits, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=True,
                   sync_dist=True, reduce_fx=torch.mean)
        result.log('val_acc', acc, prog_bar=True, 
                   sync_dist=True, reduce_fx=torch.mean)
        return result
        
    def test_step(self, batch, batch_index):
        x, y, i = batch
        pred = self.net(x).argmax(1)
        return {'pred':pred, 'label':y, 'index':i}
    
    def test_step_end(self, outputs):
        if isinstance(outputs, dict):
            return outputs
        keys = list(outputs[0].keys())
        combined = {}
        for k in keys:
            c = torch.cat([x[k] for x in outputs])
            combined[k] = c
        return combined
    
    def test_epoch_end(self, results):
        keys = list(results[0].keys())
        combined = {}
        for k in keys:
            c = torch.cat([x[k] for x in results]).cpu()
            combined[k] = c
        dset = self.trainer.test_dataloaders[0].dataset
        paths = [dset.imgs[i] for i in combined['index'].tolist()]
        combined['paths'] = paths
        torch.save(combined, self.output_file)
        return {'result': 1}
    
    
class CastleRetrieval(CastleClassifier):
    '''LightningModule for castle image retrieval'''
    def _set_net(self, hp):
        self.net = resnet(self.nclass, hp.net, classifier=False)
        
    def _set_lossfn(self, hp):
        from pytorch_metric_learning import losses as pmloss
        from pytorch_metric_learning import regularizers as pmreg
        from pytorch_metric_learning import miners as pmminers

        lossfn = getattr(pmloss, hp.loss, None)
        if not lossfn:
            raise AttributeError(f'Could not find loss function {loss}')
        self.lossfn = lossfn(margin=hp.margin,
                             embedding_regularizer=pmreg.LpRegularizer())
        self.miner = pmminers.MultiSimilarityMiner()
    
    @staticmethod
    def add_model_specific_args(parent):
        parser = ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--data', type=str, default=dataset.DATA_ROOT)
        parser.add_argument('--net', type=str, default='resnet50')
        parser.add_argument('--optim', type=str, default='Adam')
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--bs', type=int, default=32)
        parser.add_argument('--workers', type=int, default=4)
        parser.add_argument('--loss', type=str, default='TripletMarginLoss')
        parser.add_argument('--margin', type=float, default=0.2)
        parser.add_argument('--npos', type=int, default=4)
        
        return parser
    
    def training_step(self, batch, batch_index):
        x, y = batch
        embed = self.net(x)
        hard_pairs = self.miner(embed, y)
        loss = self.lossfn(embed, y, hard_pairs)
        acc = map_r(embed, y, self.hp.npos)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        result.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return result
    
    def validation_step(self, batch, batch_index):
        x, y = batch
        embed = self.net(x)
        hard_pairs = self.miner(embed, y)
        loss = self.lossfn(embed, y, hard_pairs)
        acc = map_r(embed, y, self.hp.npos)
        result = pl.EvalResult(checkpoint_on=acc)
        result.log('val_loss', loss, prog_bar=True,
                   sync_dist=True, reduce_fx=torch.mean)
        result.log('val_acc', acc, prog_bar=True,
                   sync_dist=True, reduce_fx=torch.mean)
        return result
            
    def test_step(self, batch, batch_index):
        x, y, i = batch
        embed = self.net(x)
        return {'pred':embed, 'label':y, 'index':i}
    
    
class CastleClassTester(CastleClassifier):
    '''LightningModule for computing various classification metrics at test time'''
    def set_label_map(self, label_map):
        # label map should be a 1d array-like of length `nclass`, where each
        # entry specifies the new label (duplicates are allowed)
        self.label_map = torch.as_tensor(label_map)
        
    def test_step(self, batch, batch_index):
        x, y = batch
        logits = self.net(x)
        return {'logits':logits, 'y':y}
    
    def test_step_end(self, output_parts):
        if isinstance(output_parts, dict):
            return output_parts
        logits = torch.cat([x[0] for x in output_parts], 0)
        y = torch.cat([x[1] for x in output_parts])
        return {'logits':logits, 'y':y}
    
    def test_epoch_end(self, results):
        logits = torch.cat([x['logits'] for x in results],0)
        y = torch.cat([x['y'] for x in results])
        # loss
        loss = self.lossfn(logits, y)
        # accuracy
        acc = calc_accuracy(logits, y)
        # mean per-class accuracy
        yhat = logits.argmax(1)
        nc = logits.shape[1]
        correct = yhat.eq(y).float()
        pcacc = torch.zeros(nc,device=y.device)\
                .scatter_add_(0,y,correct)\
                .div_(torch.bincount(y,None,nc)).mean()
        # tax accuracy
        label_map = self.label_map.to(y)
        yy = label_map[y]
        xx = label_map[yhat]
        tax_acc = xx.eq(yy).float().mean()
        # mean per-class tax accuracy
        nc = label_map.max()+1
        correct = xx.eq(yy).float()
        pc_tax_acc = torch.zeros(nc,device=yy.device)\
                     .scatter_add_(0,yy,correct)\
                     .div(torch.bincount(yy,None,nc)).mean()
        
        results = dict(
            test_loss=loss,
            test_acc=acc,
            test_acc_per_class=pcacc,
            test_acc_tax=tax_acc,
            test_acc_tax_per_class=pc_tax_acc
        )
        self.log_vals(results)
            
        return results
    
    @pl.utilities.rank_zero_only
    def log_vals(self, v):
        self.logger.experiment.log_metrics(v)
        self.logger.experiment.save()

    
class CastleCrossClassifier(CastleClassifier):
    '''LightningModule for cross-validation. Overrides CastleClassifier to save
    individual image predictions during testing.'''
    def optimizer_step(self, current_epoch, batch_nb, optimizer,
                       optimizer_idx, second_order_closure=None, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        # linear warmup to flat schedule
        total = self.trainer.max_epochs * len(self.trainer.train_dataloader)
        cut = int(0.3 * total)
        if self.trainer.global_step < cut:
            lr_scale = min(1., float(self.trainer.global_step + 1) / cut)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hp.lr

        optimizer.step()
        optimizer.zero_grad()
        
    def test_step(self, batch, batch_idx):
        x, y, i = batch
        logits = self.net(x)
        return {'preds':logits, 'labels':y, 'inds':i}
    
    def test_epoch_end(self, results):
        preds = torch.cat([x['preds'] for x in results], 0).argmax(1)
        labels = torch.cat([x['labels'] for x in results], 0)
        inds = torch.cat([x['inds'] for x in results], 0)
        self.preds, self.labels, self.inds = preds, labels, inds
        return {}

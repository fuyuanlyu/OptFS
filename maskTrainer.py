import torch
import argparse
import logging
import os, sys
from pathlib import Path
import numpy as np
from sklearn import metrics
from utils import trainUtils
from modules import mask

parser = argparse.ArgumentParser(description="optfs trainer")
parser.add_argument("dataset", type=str, help="specify dataset")
parser.add_argument("model", type=str, help="specify model")

# dataset information
parser.add_argument("--feature", type=int, help="feature number", required=True)
parser.add_argument("--field", type=int, help="field number", required=True)
parser.add_argument("--data_dir", type=str, help="data directory", required=True)

# training hyperparameters
parser.add_argument("--lr", type=float, help="learning rate" , default=1e-4)
parser.add_argument("--l2", type=float, help="L2 regularization", default=1e-5)
parser.add_argument("--bsize", type=int, help="batchsize", default=4096)
parser.add_argument("--optim", type=str, default="Adam", help="optimizer type")
parser.add_argument("--max_epoch", type=int, default=15, help="maxmium epochs")
parser.add_argument("--save_dir", type=Path, help="model save directory")

# neural network hyperparameters
parser.add_argument("--dim", type=int, help="embedding dimension", default=16)
parser.add_argument("--mlp_dims", type=int, nargs='+', default=[1024, 512, 256], help="mlp layer size")
parser.add_argument("--mlp_dropout", type=float, default=0.0, help="mlp dropout rate (default:0.0)")
parser.add_argument("--mlp_bn", action="store_true", help="mlp batch normalization")
parser.add_argument("--cross", type=int, help="cross layer", default=3)

# device information
parser.add_argument("--cuda", type=int, choices=range(-1, 8), default=-1, help="device info")

# mask information
parser.add_argument("--mask_init", type=float, default=0.5, help="mask initial value" )
parser.add_argument("--final_temp", type=float, default=200.0, help="final temperature")
parser.add_argument("--search_epoch", type=int, default=20, help="search epochs")
parser.add_argument("--rewind_epoch", type=int, default=1, help="rewind epoch")
parser.add_argument("--reg_lambda", type=float, default=1e-8, help="regularization rate")
args = parser.parse_args()

my_seed = 2022
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
np.random.seed(my_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'

class Trainer(object):
    def __init__(self, opt):
        self.lr = opt['lr']
        self.l2 = opt['l2']
        self.bs = opt['bsize']
        self.model_dir = opt["save_dir"]
        self.epochs = opt["search_epoch"]
        self.rewind_epoch = opt["rewind_epoch"]
        self.reg_lambda = opt["lambda"]
        self.temp_increase = opt["final_temp"] ** (1./ (opt["search_epoch"]-1))
        self.dataloader = trainUtils.getDataLoader(opt["dataset"], opt["data_dir"])
        self.device = trainUtils.getDevice(opt["cuda"])
        self.network = mask.getModel(opt["model"],opt["model_opt"]).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optim = mask.getOptim(self.network, opt["optimizer"],self.lr, self.l2)
    
    def train_on_batch(self, label, data, retrain=False):
        self.network.train()
        self.network.zero_grad()
        data, label = data.to(self.device), label.to(self.device)
        logit = self.network(data)
        logloss = self.criterion(logit, label)
        regloss = self.network.reg()
        if not retrain:
            loss = self.reg_lambda * regloss + logloss
        else:
            loss = logloss
        loss.backward()
        for optim in self.optim:
            optim.step()
        return logloss.item()
    
    def eval_on_batch(self, data):
        self.network.eval()
        with torch.no_grad():
            data = data.to(self.device)
            logit = self.network(data)
            prob = torch.sigmoid(logit).detach().cpu().numpy()
        return prob
    
    def search(self):
        print("ticket:{t}".format(t=self.network.ticket))
        print("-----------------Begin Search-----------------")
        for epoch_idx in range(int(self.epochs)):
            train_loss = .0
            step = 0
            if epoch_idx > 0:
                self.network.temp *= self.temp_increase
            if epoch_idx == self.rewind_epoch:
                self.network.checkpoint()
            for feature, label in self.dataloader.get_data("train", batch_size = self.bs):
                train_loss += self.train_on_batch(label, feature)
                step += 1
            train_loss /= step
            print("Temp:{temp:.6f}".format(temp=self.network.temp))
            val_auc, val_loss = self.evaluate("val")
            print("[Epoch {epoch:d} | Train Loss: {loss:.6f} | Val AUC: {val_auc:.6f}, Val Loss: {val_loss:.6f}]".format(epoch=epoch_idx,  loss=train_loss, val_auc=val_auc, val_loss=val_loss))
            rate = self.network.compute_remaining_weights()
            print("Feature remain:{rate:.6f}".format(rate=rate))
        test_auc, test_loss = self.evaluate("test")
        print("Test AUC: {test_auc:.6f}, Test Loss: {test_loss:.6f}".format(test_auc=test_auc, test_loss=test_loss))

    def evaluate(self, on:str):
        preds, trues = [], []
        for feature, label in self.dataloader.get_data(on, batch_size=self.bs * 10):
            pred = self.eval_on_batch(feature)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss
    
    def train(self, epochs):
        self.network.ticket=True
        self.network.rewind_weights()
        cur_auc = 0.0
        early_stop = False
        self.optim = mask.getOptim(self.network, "adam", self.lr, self.l2)[:1]
        rate = self.network.compute_remaining_weights()
    
        print("-----------------Begin Train-----------------")
        print("Ticket:{t}".format(t=self.network.ticket))
        print("Final feature remain:{rate:.6f}".format(rate=rate))
        for epoch_idx in range(int(epochs)):
            train_loss = .0
            step = 0
            for feature, label in self.dataloader.get_data("train", batch_size = self.bs):
                train_loss += self.train_on_batch(label, feature, retrain=True)
                step += 1
            train_loss /= step
            val_auc, val_loss = self.evaluate("val")
            print("[Epoch {epoch:d} | Train Loss:{loss:.6f} | Val AUC:{val_auc:.6f}, Val Loss:{val_loss:.6f}]".format(epoch=epoch_idx, loss=train_loss, val_auc=val_auc, val_loss=val_loss))
            
            if val_auc > cur_auc:
                cur_auc = val_auc
                torch.save(self.network.state_dict(), self.model_dir)
            else:
                self.network.load_state_dict(torch.load(self.model_dir))
                self.network.to(self.device)
                early_stop = True
                test_auc, test_loss = self.evaluate("test")
                print("Early stop at epoch {epoch:d} | Test AUC: {test_auc:.6f}, Test Loss:{test_loss:.6f}".format(epoch=epoch_idx, test_auc = test_auc, test_loss = test_loss))
                break
        
        if not early_stop:
            test_auc, test_loss = self.evaluate("test")
            print("Final Test AUC: {test_auc:.6f}, Test Loss: {test_loss:.6f}".format(test_auc=test_auc, test_loss=test_loss))

def main():
    model_opt={
        "latent_dim":args.dim, "feat_num":args.feature, "field_num":args.field, 
        "mlp_dropout":args.mlp_dropout, "use_bn": args.mlp_bn, "mlp_dims":args.mlp_dims,
        "mask_initial":args.mask_init,"cross":args.cross
        }

    opt={
        "model_opt":model_opt, "dataset":args.dataset, "model":args.model, "lr":args.lr, "l2":args.l2,
        "bsize":args.bsize, "optimizer":args.optim, "data_dir":args.data_dir,"save_dir":args.save_dir, 
        "cuda":args.cuda,"search_epoch":args.search_epoch, "rewind_epoch": args.rewind_epoch,"final_temp":args.final_temp,
        "lambda":args.reg_lambda
        }
    print(opt)
    trainer = Trainer(opt)
    trainer.search()
    trainer.train(args.max_epoch)
    
if __name__ == "__main__":
    """
    python trainer.py Criteo DeepFM --feature    
    """
    main()

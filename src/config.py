import os
import time
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from load_data import *
from model import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error

class Logger:

    def __init__(self, path, log=True):
        self.path = path
        self.log = log

    def __call__(self, content, **kwargs):
        print(content, **kwargs)
        if self.log:
            with open(self.path, 'a') as f:
                print(content, file=f, **kwargs)

class Config:
    
    def __init__(self, data="housing", model="NAE", lr = 1e-4, max_epoch = 50, batch_size=256, \
         test_step=1, h_dim=64, n_layers=4, device="cpu", seed=0, exp_str=None, eval=False, \
         weight_decay = 1e-8, dropout=0.0, dropout_expert=0.0, fold=0, output_penalty=0.0, \
         var_penalty=0.1, C=32, k=2, batch_norm=False, DATA_PATH = "./data"):
    
        assert fold >= 0 and fold < 5

        self.fold = fold
        self.seed = seed
        self.eval = eval

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.data = data
        # load the target data
        if data.lower() == "mimic2":
            dataset = load_mimic2(fold=fold, DATA_PATH=DATA_PATH)
        elif data.lower() == "mimic3":
            dataset = load_mimic3(fold=fold, DATA_PATH=DATA_PATH)
        elif data.lower() == "credit":
            dataset = load_credit(fold=fold, DATA_PATH=DATA_PATH)
        elif data.lower() == "income":
            dataset = load_income(fold=fold, DATA_PATH=DATA_PATH)
        elif data.lower() == "housing":
            dataset = load_housing()
        elif data.lower() == "year":
            dataset = load_year(DATA_PATH=DATA_PATH)
        else:
            raise ValueError("Dataset {:s} not supported".format(data))
        
        self.preprocessor = Preprocessor(
            cat_features=dataset.get('cat_features', None),
            y_normalize=(dataset['problem'] == 'regression'),
            random_state=self.seed,
            quantile_transform=True,
            quantile_noise=1e-3,
            n_quantiles=2000,
            output_distribution='normal'
        )

        X_train, y_train = dataset['X_train'], dataset['y_train']
        X_test, y_test = dataset['X_test'], dataset['y_test']

        X_val = dataset.get('X_val', None)
        if X_val is not None:
            y_val = dataset['y_val']
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,\
                 random_state=self.seed, stratify=(y_train if dataset['problem'] == 'classification' else None))

        self.preprocessor.fit(X_train, y_train)
        X_train, y_train = self.preprocessor.transform(X_train, y_train)
        X_val, y_val = self.preprocessor.transform(X_val, y_val)
        X_test, y_test = self.preprocessor.transform(X_test, y_test)

        self.train_set = OurDataset(X_train, y_train)
        self.val_set = OurDataset(X_val, y_val)
        self.test_set = OurDataset(X_test, y_test)

        self.weight_decay = weight_decay
        self.model_type = model
        self.problem = dataset["problem"]

        self.device = device
        self.lr = lr
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.train_batch_size = self.batch_size        
        self.test_step = test_step
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.C = C
        self.k = k
        self.batch_norm = batch_norm
        if self.problem == "regression":
            self.n_class = 1
        else: 
            self.n_class = max(self.train_set.y) + 1
        self.dropout = dropout
        self.dropout_expert = dropout_expert
        self.output_penalty = output_penalty
        self.var_penalty = var_penalty

        if self.model_type == "NAE":
            self.model = NAE(self.problem, len(self.train_set.X[0]), self.h_dim, \
                self.n_layers, 1 if self.n_class < 3 else self.n_class, C=self.C, k=self.k, \
                dropout=self.dropout, dropout_expert=self.dropout_expert, output_penalty=self.output_penalty, \
                var_penalty=self.var_penalty, batch_norm=self.batch_norm)
        elif self.model_type == "NAE_D":
            self.model = NAE_D(self.problem, len(self.train_set.X[0]), self.h_dim, \
                self.n_layers, 1 if self.n_class < 3 else self.n_class, C=self.C, k=self.k, \
                dropout=self.dropout, dropout_expert=self.dropout_expert, output_penalty=self.output_penalty, \
                var_penalty=self.var_penalty, batch_norm=self.batch_norm)        
        elif self.model_type == "NAE_E":
            self.model = NAE_E(self.problem, len(self.train_set.X[0]), self.h_dim, \
                self.n_layers, 1 if self.n_class < 3 else self.n_class, C=self.C, k=self.k, \
                dropout=self.dropout, dropout_expert=self.dropout_expert, output_penalty=self.output_penalty, \
                var_penalty=self.var_penalty, batch_norm=self.batch_norm)
        else:
            raise ValueError("Model {:s} not supported".format(self.model_type))

        assert exp_str is not None

        self.checkpoint_dir = os.path.join("./checkpoint", data, self.model_type, exp_str)
        self.log_dir = os.path.join("./log", data, self.model_type, exp_str)

        if self.eval:
            self.logger = Logger(os.path.join(self.log_dir, "log.txt"), log=False)
            self.tf_writer = lambda *_, **__: None
            self.tf_writer.add_scalar = lambda *_, **__: None
        else:
            self.logger = Logger(os.path.join(self.log_dir, "log.txt"))
            self.tf_writer = SummaryWriter(log_dir=self.log_dir)
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

        self.model.to(self.device)
        
        self.logger("*" * 40)
        self.logger("Number of hidden layers: {:d}".format(self.n_layers))
        self.logger("Dimension of hidden layer: {:d}".format(self.h_dim))
        self.logger("Number of total experts: {:d}".format(self.C))
        self.logger("Number of activated experts: {:d}".format(self.k))
        self.logger("Max epoch: {:d}".format(self.max_epoch))
        self.logger("Batch size: {:d}".format(self.batch_size))
        self.logger("Learning rate: {:f}".format(self.lr))
        self.logger("Weight decay: {:f}".format(self.weight_decay))
        self.logger("Output penalty: {:f}".format(self.output_penalty))
        self.logger("Variation penalty: {:f}".format(self.var_penalty))
        self.logger("Dropout: {:f}".format(self.dropout))
        self.logger("Dropout expert: {:f}".format(self.dropout_expert))
        self.logger("Test step: {:d}".format(self.test_step))
        self.logger("Device: {:s}".format(self.device))
        self.logger("Fold: {}".format(self.fold))
        self.logger("Seed: {}".format(self.seed))
        self.logger("*" * 40)
        self.logger("")

    def train(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.seed)

        # if self.device == "cuda":
        #     train_loader = DataLoader(self.train_set, batch_size = self.train_batch_size, shuffle = True, num_workers=1, pin_memory=True) # num_workers=1
        # else:
        train_loader = DataLoader(self.train_set, batch_size = self.train_batch_size, shuffle = True, pin_memory=True)

        self.logger("Training Starts\n")
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer = optim, T_max = self.max_epoch
        )
        self.train_session(train_loader, self.max_epoch, optim, scheduler, device=self.device)

        self.logger("Training Ends\n")

        self.logger("[Evaluation on Test Set]")
        start_time = time.time()
        test_loss, score_dict_test = self.evaluate(dataset="test", \
                checkpoint_path=os.path.join(self.checkpoint_dir, "best_model.pt"), device=self.device)
        
        self.logger(" | ".join(["Time: %.1fs" % (time.time() - start_time), \
                    "loss/test: %.4f" % (test_loss / len(self.test_set))] + \
                    ["%s: %.4f" % (key, value) for key, value in score_dict_test.items()]))

        self.tf_writer.add_scalar("loss/test", test_loss / len(self.test_set), self.max_epoch)
        for key, value in score_dict_test.items():
            self.tf_writer.add_scalar(key, value, self.max_epoch)

        self.logger("")

    def train_session(self, train_loader, max_epoch, optim, scheduler=None, device="cpu"):

        best_epoch = 0
        best_metric = -1e9

        for epoch in range(max_epoch):

            self.model.train()

            self.logger("[Epoch {:d}]".format(epoch))
            train_loss = 0
            start_time = time.time()
            y_truth = []
            y_logits = []

            for bat in train_loader:
                x = bat[0]
                y = bat[1]
                x = x.float().to(device)
                y = y.float().to(device)                
                optim.zero_grad()                
                loss, logits = self.model(x, y)
                
                loss.backward()
            
                optim.step()
                train_loss += loss.item() * len(x)
                y_truth.append(y)
                y_logits.append(logits.detach())

            if scheduler is not None:
                scheduler.step()

            y_truth = torch.cat(y_truth)
            y_logits = torch.cat(y_logits)

            if self.problem == "regression":
                y_truth = y_truth.cpu() * self.preprocessor.y_std + self.preprocessor.y_mu
                y_logits = y_logits.cpu() * self.preprocessor.y_std + self.preprocessor.y_mu
                train_mse = mean_squared_error(y_truth, y_logits)
                train_rmse = mean_squared_error(y_truth, y_logits, squared=False)                
                score_dict_train = {"mse/train": train_mse, "rmse/train": train_rmse}
            else:
                if self.n_class == 2:
                    y_logits = torch.sigmoid(y_logits)
                    y_pred = (y_logits > 0.5).long()
                    train_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu())
                else:
                    y_logits = torch.softmax(y_logits, dim=-1)
                    y_pred = y_logits.argmax(dim=-1)                
                    train_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu(), multi_class="ovo")
                train_acc = accuracy_score(y_truth.cpu(), y_pred.cpu())
                train_f1 = f1_score(y_truth.cpu(), y_pred.cpu(), average="macro")
                score_dict_train = {"accuracy/train": train_acc, "auc/train": train_auc, "f1/train": train_f1}

            self.logger(" | ".join(["Time: %.1fs" % (time.time() - start_time), \
                        "loss/train: %.4f" % (train_loss / len(self.train_set))] + \
                        ["%s: %.4f" % (key, value) for key, value in score_dict_train.items()]))
            self.tf_writer.add_scalar("loss/train", train_loss / len(self.train_set), epoch)
            for key, value in score_dict_train.items():
                self.tf_writer.add_scalar(key, value, epoch)

            if (epoch + 1) % self.test_step == 0:
                self.model.eval()

                self.logger("[Evaluation on Validation Set]")
                start_time = time.time()

                val_loss, score_dict_val = self.evaluate(dataset="val", device=device)

                if self.problem == "regression":
                    curr_metric = - score_dict_val["rmse/val"]
                else:
                    curr_metric = score_dict_val["auc/val"] + 0.1 * score_dict_val["f1/val"]# + 0.001 * score_dict_train["auc/train"]


                self.logger(" | ".join(["Time: %.1fs" % (time.time() - start_time), \
                            "loss/val: %.4f" % (val_loss / len(self.val_set))] + \
                            ["%s: %.4f" % (key, value) for key, value in score_dict_val.items()]))

                self.tf_writer.add_scalar("loss/val", val_loss / len(self.val_set), epoch)
                for key, value in score_dict_val.items():
                    self.tf_writer.add_scalar(key, value, epoch)

                if curr_metric > best_metric:
                    torch.save(self.model.state_dict(), \
                        os.path.join(self.checkpoint_dir, "best_model.pt"))
                    best_metric = curr_metric
                    best_epoch = epoch
                    self.logger("Model Saved!")
                    
            self.logger("")

            if epoch - best_epoch > max_epoch // 4:
                break
                    
        self.logger("Best epoch: {:d}\n".format(best_epoch))

    def evaluate(self, dataset="test", checkpoint_path=None, device="cpu"):

        self.model.eval()

        if checkpoint_path is not None:
            print("Loading checkpoint from:", checkpoint_path)
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        if dataset == "train":
            loader = DataLoader(self.train_set, batch_size = self.batch_size, shuffle = False)
        elif dataset == "val":
            loader = DataLoader(self.val_set, batch_size = self.batch_size, shuffle = False)
        elif dataset == "test":
            loader = DataLoader(self.test_set, batch_size = self.batch_size, shuffle = False)

        total_loss = 0

        with torch.no_grad():
            
            y_truth = []
            y_logits = []

            for bat in loader:
                x = bat[0]
                y = bat[1]
                x = x.float().to(device)
                y = y.float().to(device)
                loss, logits = self.model(x, y)
                loss = loss.mean()

                total_loss += loss.item() * len(x)

                y_truth.append(y)
                y_logits.append(logits.detach())
            
            y_truth = torch.cat(y_truth)
            y_logits = torch.cat(y_logits)

            if self.problem == "regression":
                y_truth = y_truth.cpu() * self.preprocessor.y_std + self.preprocessor.y_mu
                y_logits = y_logits.cpu() * self.preprocessor.y_std + self.preprocessor.y_mu
                mse = mean_squared_error(y_truth, y_logits)
                rmse = mean_squared_error(y_truth, y_logits, squared=False)
                score_dict = {"mse/"+dataset: mse, "rmse/"+dataset: rmse}
            else:
                if self.n_class == 2:
                    y_logits = torch.sigmoid(y_logits)
                    y_pred = (y_logits > 0.5).long()
                    auc = roc_auc_score(y_truth.cpu(), y_logits.cpu())
                else:
                    y_logits = torch.softmax(y_logits, dim=-1)
                    y_pred = y_logits.argmax(dim=-1)                
                    auc = roc_auc_score(y_truth.cpu(), y_logits.cpu(), multi_class="ovo")
                acc = accuracy_score(y_truth.cpu(), y_pred.cpu())
                f1 = f1_score(y_truth.cpu(), y_pred.cpu(), average="macro")
                score_dict = {"accuracy/"+dataset: acc, "auc/"+dataset: auc, "f1/"+dataset: f1}

            return total_loss, score_dict
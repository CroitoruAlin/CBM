import os

import torch
from timm.scheduler import create_scheduler

from test import test
from cvt_base_pretrained import Trainer


class TrainerCvt(Trainer):
    def __init__(self, model, train_loader, val_loader,add_optimizer_params_lr, args,build_optimizer, config):
        super().__init__(model, train_loader, val_loader,add_optimizer_params_lr, args,build_optimizer)
        attr_dict = AttrDict()
        config["MODEL"]["TRAIN"]["LR_SCHEDULER"]["ARGS"]['epochs'] = args.num_epochs
        attr_dict.update(config["MODEL"]["TRAIN"]["LR_SCHEDULER"]["ARGS"])
        config["MODEL"]["TRAIN"]["LR_SCHEDULER"]["ARGS"] = attr_dict
        self.lr_scheduler = None
        self.config=config

    def train(self):
        best_accuracy = 0.
        for epoch in range(1, self.args.num_epochs + 1):
            print('Epoch', epoch)
            super(TrainerCvt, self)._train_epoch(epoch-1)

            test_acc = test(self.model, self.args.device, self.val_loader)
            test(self.model, self.args.device,self.args.testlo)
            if best_accuracy < test_acc:
                best_accuracy = test_acc
                torch.save(self.model.state_dict(), os.path.join( "saved_models", self.args.model_name + ".pth"))
                print('New best accuracy. Model Saved!')
        return best_accuracy

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self





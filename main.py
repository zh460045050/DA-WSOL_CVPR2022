import numpy as np
import os
import random
import torch
import torch.nn as nn
from libs.solver import Trainer

def main():
    trainer = Trainer()

    print("===========================================================")

    if trainer.args.check_path != "":
        checkpoint = torch.load(trainer.args.check_path)
        trainer.extractor.load_state_dict(checkpoint['state_dict_extractor'], strict=True)
        trainer.classifier.load_state_dict(checkpoint['state_dict_classifier'], strict=True)
        if trainer.args.wsol_method == 'dann':
            trainer.domain_classifier.load_state_dict(checkpoint['state_dict_domain_classifier'], strict=True)

    if trainer.args.mode == "training":

        for epoch in range(trainer.args.epochs):
            trainer.epoch = epoch
            print("===========================================================")
            print("Start epoch {} ...".format(epoch + 1))
            trainer.adjust_learning_rate(epoch + 1)
            train_performance = trainer.train(split='train')
            #print(train_performance)
            trainer.report_train(train_performance, epoch + 1, split='train')
            if (epoch + 1) % trainer.args.eval_frequency == 0:
                trainer.evaluate(epoch=epoch + 1, split='val')
                trainer.print_performances()
                trainer.report(epoch=epoch + 1, split='val')
                trainer.save_checkpoint(epoch + 1, split='val')
            print("Epoch {} done.".format(epoch + 1))

        print("===========================================================")
        print("Evaluation on validation set ...")
        trainer.save_checkpoint(epoch + 1, split='val')
        trainer.load_checkpoint(checkpoint_type=trainer.args.eval_checkpoint_type)

        if trainer.args.dataset_name != "ILSVRC":
            print("===========================================================")
            print("Evaluation on test set ...")
            trainer.evaluate(epoch=epoch + 1, split='test')
            trainer.print_performances()
            trainer.report(epoch=epoch + 1, split='test')
            trainer.save_checkpoint(epoch + 1, split='test')
    elif trainer.args.mode == "test":
        trainer.epoch = 0
        #trainer.save_checkpoint(0, split='val')
        print("===========================================================")
        print("Evaluation on validation set ...")
        trainer.evaluate(epoch=0, split='val')
        trainer.print_performances()
        trainer.report(epoch=0, split='val')
        
        if trainer.args.dataset_name != "ILSVRC":
            print("===========================================================")
            print("Evaluation on test set ...")
            trainer.evaluate(epoch=0, split='test')
            trainer.print_performances()
            trainer.report(epoch=0, split='test')

    


if __name__ == '__main__':
    from torch.backends import cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True
    main()

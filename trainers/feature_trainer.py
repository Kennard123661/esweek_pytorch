import os
import sys
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from data.feature_data import esweek_training_collate_fn, esweek_inference_collate_fn
from models.feature_model import FeatureModel
from data.feature_data import TrainDataset

base_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'))

if __name__ == '__main__':
    sys.path.append(base_dir)


class FeatureTrainer():
    def __init__(self, checkpoint_name, margin=0.1, cls_loss_weighting=0.5,
                 max_epoch=200, seed=123, learning_rate=10e-4, lr_stepsize=20,
                 lr_decay=0.5):
        checkpoint_dir = os.path.join(base_dir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_name = checkpoint_name
        self.checkpoint_filepath = os.path.join(checkpoint_dir,
                                                '{}.pth'.format(self.checkpoint_name))
        self.margin = margin
        self.model = FeatureModel()
        self.seed = seed
        torch.random.manual_seed(seed)
        self.max_epoch = max_epoch
        self.epoch = 0

        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin)
        self.classification_loss = nn.CrossEntropyLoss()
        self.cls_loss_weighting = float(cls_loss_weighting)
        assert self.cls_loss_weighting >= 0 and self.cls_loss_weighting <= 1

        self.learning_rate = learning_rate
        self.lr_stepsize = lr_stepsize
        self.lr_decay = lr_decay
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optim_scheduler =  optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_stepsize,
                                                          gamma=self.lr_decay)
        self.load_checkpoint(self.checkpoint_filepath)

    def save_checkpoint(self, checkpoint_filepath):
        print("saving checkpoint...")
        model_state = self.model.state_dict()
        epoch = self.epoch
        dict_to_save = {'model': model_state,
                        'epoch': epoch,
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.optim_scheduler.state_dict()}
        torch.save(dict_to_save, checkpoint_filepath)

    def load_checkpoint(self, checkpoint_filepath):
        if os.path.exists(checkpoint_filepath):
            print("checkpoint exists, loading...")
            checkpoint = torch.load(checkpoint_filepath)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.optim_scheduler.load_state_dict(checkpoint['scheduler'])
            self.epoch = checkpoint['epoch']

    def train(self, train_dataset, val_dataset, batch_size, eval_batchsize, num_workers=4, save_every=5):
        assert isinstance(train_dataset, TrainDataset)
        train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers,
                                           collate_fn=esweek_training_collate_fn)
        eval_dataloader = data.DataLoader(dataset=val_dataset, batch_size=eval_batchsize,
                                          shuffle=True, num_workers=num_workers,
                                          collate_fn=esweek_inference_collate_fn)
        self.num_eval = len(val_dataset)
        while self.epoch < self.max_epoch:
            self.epoch += 1
            self.optim_scheduler.step()
            self.train_step(train_dataloader)
            if self.epoch > 0 and self.epoch % save_every == 0:
                self.save_checkpoint(self.checkpoint_filepath)
            self.eval_step(eval_dataloader)
        self.eval_step(eval_dataloader)

    def train_step(self, train_dataloader):
        epoch_loss = 0
        for (anchor_feats, positive_feats, negative_feats,
             anchor_labels, positive_labels, negative_labels) in train_dataloader:
            self.optimizer.zero_grad()
            anchor_outs = self.model(anchor_feats)
            positive_outs = self.model(positive_feats)
            negative_outs = self.model(negative_feats)
            anchor_intermediates, anchor_preds = anchor_outs[:-1], anchor_outs[-1]
            positive_intermediates, positive_preds = positive_outs[:-1], positive_outs[-1]
            negative_intermediates, negative_preds = negative_outs[:-1], negative_outs[-1]

            triplet_loss = 0
            num_intermediates = len(anchor_intermediates)
            num_anchors = anchor_preds.shape[0]
            triplet_loss_divisor = 0
            for i in range(num_intermediates):
                anchors = anchor_intermediates[i].view(num_anchors, -1)
                positives = positive_intermediates[i].view(num_anchors, -1)
                negatives = negative_intermediates[i].view(num_anchors, -1)
                normalized_margin_loss = self.normalized_triplet_margin_loss(anchors, positives, negatives) * (i + 1)
                normalized_margin_loss = torch.clamp(normalized_margin_loss, min=0.0)
                triplet_loss += normalized_margin_loss
                triplet_loss_divisor += (i + 1)
            triplet_loss /= triplet_loss_divisor

            anchor_cls_loss = self.classification_loss(anchors, anchor_labels.view(-1))
            positive_cls_loss = self.classification_loss(positives, positive_labels.view(-1))
            negative_cls_loss = self.classification_loss(negatives, negative_labels.view(-1))
            cls_loss = (anchor_cls_loss + positive_cls_loss + negative_cls_loss) / 3

            total_loss = self.cls_loss_weighting * cls_loss + (1 - self.cls_loss_weighting) * cls_loss
            total_loss.backward()
            self.optimizer.step()
            epoch_loss += total_loss.item()
        # print("epoch {0}, loss: {1}".format(self.epoch, epoch_loss))

    def eval_step(self, eval_dataloader):
        self.model.eval()
        accuracy = 0
        for (feats, labels) in eval_dataloader:
            predictions = self.model(feats)[-1]
            predictions = torch.argmax(predictions, dim=1)

            predictions = np.array(predictions.view(-1).detach().tolist())
            labels = np.array(labels.view(-1).tolist())
            is_correct = np.equal(predictions, labels).astype(int)
            num_correct = np.sum(is_correct)
            accuracy += num_correct
        accuracy /= self.num_eval
        print("epoch {0}... The accuracy is {1}".format(self.epoch, accuracy))

    def normalized_triplet_margin_loss(self, anchors, positives, negatives):
        normed_anchors = F.normalize(anchors)
        normed_positives = F.normalize(positives)
        normed_negatives = F.normalize(negatives)
        return self.triplet_loss(normed_anchors, normed_positives, normed_negatives)
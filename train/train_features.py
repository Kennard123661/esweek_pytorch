import os
import sys

from data.feature_data import TrainDataset, InferenceDataset
from trainers.feature_trainer import FeatureTrainer

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..'))
    sys.path.append(base_dir)



if __name__ == '__main__':
    trainer = FeatureTrainer(checkpoint_name='draft', max_epoch=1)
    train_dataset = TrainDataset()
    inference_dataset = InferenceDataset(data='test')
    trainer.train(train_dataset, inference_dataset, batch_size=int(len(train_dataset) / 3),
                  eval_batchsize=len(inference_dataset), save_every=1)

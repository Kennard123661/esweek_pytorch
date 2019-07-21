import os
import sys
import torch

import numpy as np
from scipy.io import loadmat

from torch.utils import data
from tqdm import tqdm

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..'))
    sys.path.append(base_dir)

ESWEEK_DIR = '/mnt/sda2/kennard/data/esweek_dataset'
BASELINE_CLASSIFIER_DIR = os.path.join(ESWEEK_DIR, 'baseline_classifier')
ESWEEK_FEATURE_FILEPATH = os.path.join(BASELINE_CLASSIFIER_DIR, 'features_file.mat')

NUM_FEATURES = 120
FEATURE_LABEL_IDX = 120
FEATURE_USER_ID = 121


def _get_features_labels_ids(features, labels, user_ids, idxs):
    idxs = idxs.reshape(-1)
    selected_features = features[idxs]
    selected_labels = labels[idxs]
    selected_user_ids = user_ids[idxs]
    return {'features': selected_features,
            'labels': selected_labels,
            'ids': selected_user_ids}


def read_feature_file():
    raw_data = loadmat(ESWEEK_FEATURE_FILEPATH)
    data = np.array(raw_data['feature_matrix_norm'], dtype=np.float)
    features = data[:, :FEATURE_LABEL_IDX]
    labels = (data[:, FEATURE_LABEL_IDX:FEATURE_LABEL_IDX+1]).astype(int) - 1
    user_ids = (data[:, FEATURE_USER_ID:FEATURE_USER_ID+1]).astype(int)
    print(np.unique(labels))

    test_idxs = np.array(raw_data['test_idx'], dtype=int)
    val_idxs = np.array(raw_data['xval_idx'], dtype=int)
    train_idxs = np.arange(data.shape[0])
    train_idxs = np.setdiff1d(np.setdiff1d(train_idxs, test_idxs.reshape(-1)), val_idxs.reshape(-1)).reshape(-1, 1)

    train_data = _get_features_labels_ids(features, labels, user_ids, train_idxs)
    val_data = _get_features_labels_ids(features, labels, user_ids, val_idxs)
    test_data = _get_features_labels_ids(features, labels, user_ids, test_idxs)
    return train_data, val_data, test_data


def esweek_training_collate_fn(batch):
    """ Creates mini-batch tensros from the
    list of tuples (query, positive, negatives)"""
    query_feats, positive_feats, negative_feats, query_labels, positive_labels, negative_labels = zip(*list(batch))

    query_feats = data.dataloader.default_collate(query_feats)
    query_labels = data.dataloader.default_collate(query_labels)
    positive_feats = data.dataloader.default_collate(positive_feats)
    positive_labels = data.dataloader.default_collate(positive_labels)
    negative_feats = data.dataloader.default_collate(negative_feats)
    negative_labels = data.dataloader.default_collate(negative_labels)
    return query_feats, positive_feats, negative_feats, query_labels, positive_labels, negative_labels


class TrainDataset(data.Dataset):
    def __init__(self):
        super(TrainDataset, self).__init__()
        self.data, _, _ = read_feature_file()
        self.anchor_idxs = np.arange(len(self.data['labels']))
        self.positive_idxs = self._get_positive_idxs()
        self.negative_idxs = self._get_negative_idxs()
        self.features = self.data['features']
        self.labels = self.data['labels']

    def _get_positive_idxs(self):
        """ Positive idxs will have the same label but different id """
        assert hasattr(self, 'data')
        labels, user_ids = self.data['labels'], self.data['ids']
        positive_idxs = list()
        all_idxs = np.arange(len(labels))
        for i, label in enumerate(tqdm(labels)):
            same_labels = np.array(labels == label).reshape(-1)
            different_ids = np.array(user_ids != user_ids[i]).reshape(-1)
            is_positives = np.bitwise_and(same_labels.reshape(-1), different_ids.reshape(-1)).reshape(-1)
            positive_idxs.append(np.array(all_idxs[is_positives]))

        # make sure there are no non-zero positive ids
        for positive_idx in positive_idxs:
            assert len(positive_idx) > 0
        return positive_idxs

    def _get_negative_idxs(self):
        labels, user_ids = self.data['labels'], self.data['ids']
        negative_idxs = list()
        all_idxs = np.arange(len(labels))
        for i, label in enumerate(tqdm(labels)):
            different_labels = np.array(labels == label).reshape(-1)
            same_ids = np.array(user_ids == user_ids[i]).reshape(-1)
            is_negatives = np.bitwise_and(different_labels, same_ids).reshape(-1)
            negative_idxs.append(all_idxs[is_negatives])

        for negative_idx in negative_idxs:
            assert len(negative_idx) > 0
        return negative_idxs

    def __len__(self):
        return len(self.anchor_idxs)

    def __getitem__(self, idx):
        idx = idx % len(self.anchor_idxs)
        positive_idx = np.random.choice(self.positive_idxs[idx], 1)
        negative_idx = np.random.choice((self.negative_idxs[idx]).reshape(-1), 1)

        anchor_feature = torch.from_numpy(self.features[idx]).float()
        positive_feature = torch.from_numpy(self.features[positive_idx]).float()
        negative_feature = torch.from_numpy(self.features[negative_idx]).float()

        anchor_label = torch.from_numpy(self.labels[idx]).long()
        positive_label = torch.from_numpy(self.labels[positive_idx]).long()
        negative_label = torch.from_numpy(self.labels[negative_idx]).long()
        return anchor_feature, positive_feature, negative_feature, \
               anchor_label, positive_label, negative_label


def esweek_inference_collate_fn(batch):
    feats, labels = zip(*list(batch))
    feats = data.dataloader.default_collate(feats)
    labels = data.dataloader.default_collate(labels)
    return feats, labels


class InferenceDataset(data.Dataset):
    def __init__(self, data):
        assert data in ['train', 'val', 'test']
        super(InferenceDataset, self).__init__()
        train, val, test = read_feature_file()
        if data == 'train':
            self.data = train
        elif data == 'val':
            self.data = val
        elif data == 'test':
            self.data = test
        else:
            raise NotImplementedError
        self.features = self.data['features']
        self.labels = self.data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        idx = idx % len(self.labels)
        feats = torch.Tensor(self.features[idx]).float()
        labels = torch.Tensor(self.labels[idx]).float()
        return feats, labels


if __name__ == '__main__':
    dataset = TrainDataset()
    print(np.array(dataset.positive_idxs))
    print(np.array(dataset.negative_idxs))



import inspect
import numpy as np
import pandas as pd
import pathlib
import sys
import torch
import json

from abc import ABCMeta, abstractmethod
from enum import Enum
from functools import lru_cache, partial
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.states import TrainerFn
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from typing import Dict, Iterable, List, Optional, Tuple

#################
import os
#################


class BaseDataset(Dataset, metaclass=ABCMeta):
    """
    Abstract base class of a dataset to be used in Hopular.
    """

    class CheckpointMode(Enum):
        """
        Enumeration of available checkpoint modes used during the training and validation of Hopular.
        """
        MIN = r'min'
        MAX = r'max'

    def encode_sample(self,
                      sample: torch.Tensor) -> torch.Tensor:
        """
        Encode all features of a single sample depending on their respective feature type.

        :param sample: sample to be encoded
        :return: encoded sample
        """
        sample = sample.view(-1)
        assert len(sample) == len(self.feature_numeric) + len(self.feature_discrete), r'Invalid sample to encode!'

        # Encode sample features according to feature type.
        sample_encoded = []
        for index, (datum, size) in enumerate(zip(sample, self.sizes)):
            if index in self.feature_numeric:
                sample_encoded.append(datum.view(-1))
            else:
                sample_encoded.append(torch.zeros(size).view(-1))
                sample_encoded[-1][datum.int().item()] = 1
        return torch.cat(sample_encoded, dim=0)

    @property
    def feature_mean(self) -> Optional[torch.Tensor]:
        return None

    @property
    def feature_stdv(self) -> Optional[torch.Tensor]:
        return None

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def sizes(self) -> Tuple[int, ...]:
        return

    @property
    @abstractmethod
    def split_train(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def split_validation(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def split_test(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def feature_numeric(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def feature_discrete(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def target_numeric(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def target_discrete(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def checkpoint_mode(self) -> CheckpointMode:
        pass


class DataModule(LightningDataModule):
    """
    Data module encapsulating a dataset to be used in Hopular, providing data loading and masking capabilities.
    """

    def __init__(self,
                 dataset: BaseDataset,
                 batch_size: Optional[int] = None,
                 super_sample_factor: int = 1,
                 noise_probability: float = 0.15,
                 mask_probability: float = 0.80,
                 replace_probability: float = 0.10,
                 num_workers: int = 0):
        """
        Initialize a data module from a dataset.

        :param dataset: dataset to be encapsulated by the data module
        :param batch_size: sample count of a single mini-batch
        :param super_sample_factor: multiplicity of the training set
        :param noise_probability: probability of selecting an input feature for the self-supervised loss
        :param mask_probability: probability of masking out a selected input feature
        :param replace_probability: probability of replacing a selected input feature with a randomly drawn feature
        :param num_workers: worker count of the data loaders
        """
        super(DataModule, self).__init__()
        self.dataset = dataset
        self.__batch_size = None if batch_size < 1 else batch_size
        self.__super_sample_factor = super_sample_factor
        self.__noise_probability = noise_probability
        self.__mask_probability = mask_probability
        self.__replace_probability = replace_probability
        self.__num_workers = num_workers
        assert 0 <= self.__noise_probability <= 1.0, r'Invalid noise probability!'
        assert 0 <= self.__mask_probability <= 1.0, r'Invalid mask probability!'
        assert 0 <= self.__replace_probability <= 1.0, r'Invalid replacement probability!'
        assert (self.__mask_probability + self.__replace_probability) <= 1.0, r'Invalid mask/replacement probabilities!'

        self.dims = self.dataset.shape[1:]
        self.memory = None
        self._has_setup_memory = False
        self.__data_train = None
        self.__data_validation = None
        self.__data_test = None

        # Register hyperparameters for logging.
        self.save_hyperparameters(ignore=[r'dataset'])

    @staticmethod
    def scale_and_noise_collate(
            samples: List[Tuple[torch.Tensor, ...]],
            mean: Optional[torch.Tensor],
            stdv: Optional[torch.Tensor],
            sizes: torch.Tensor,
            noise_probability: float,
            mask_probability: float,
            replace_probability: float,
            target_discrete: torch.Tensor,
            target_numeric: torch.Tensor,
            feature_discrete: torch.Tensor,
            exclude_targets: bool) -> Tuple[torch.Tensor, ...]:
        """
        Pre-process samples to be used in Hopular training and inference.

        :param samples: collection of samples to be pre-processed
        :param mean: feature means used for feature shifting
        :param stdv: feature standard deviations used for feature scaling
        :param sizes: feature sizes (class count for discrete features)
        :param noise_probability: probability of selecting an input feature for the self-supervised loss
        :param mask_probability: probability of masking out a selected input feature
        :param replace_probability: probability of replacing a selected input feature with a randomly drawn feature
        :param target_discrete: indices of discrete targets
        :param target_numeric: indices of continuous targets
        :param feature_discrete: indices of discrete features
        :param exclude_targets: completely mask out targets
        :return: masked samples, masking positions, unmasked samples, missing positions and original sample indices
        """
        samples_collated = {}
        for sample in samples:
            for sample_index, sample_element in enumerate(sample):
                samples_collated.setdefault(sample_index, []).append(sample_element)
        samples_collated = tuple(torch.stack(
            samples_collated[sample_index], dim=0
        ) for sample_index in sorted(samples_collated))
        feature_boundaries = torch.cumsum(torch.as_tensor([0] + sizes), dim=0)
        feature_boundaries = zip(feature_boundaries[:-1], feature_boundaries[1:])

        # Compute noise mask.
        noise_mask = torch.ones(samples_collated[0].shape[0], len(sizes))
        if noise_probability > 0:
            noise_mask = torch.dropout(noise_mask, p=1.0 - noise_probability, train=True)
            noise_mask = noise_mask != 0
        else:
            noise_mask = noise_mask == 0

        # Scale sample features according to feature statistics and add optional noise.
        samples_modified = []
        for index, (start, end) in enumerate(feature_boundaries):

            # Standardize attributes.
            if index not in feature_discrete:
                if mean is not None:
                    assert not np.isnan(mean[index])
                    samples_collated[0][:, start:end] = samples_collated[0][:, start:end] - mean[index]
                if stdv is not None and stdv[index] > 0:
                    assert not np.isnan(stdv[index])
                    samples_collated[0][:, start:end] = samples_collated[0][:, start:end] / stdv[index]

            # Encode features and targets accordingly and introduce optional noise.
            if exclude_targets and (index in target_discrete or index in target_numeric):
                current_sample = torch.cat((
                    torch.zeros(len(samples_collated[0]), end - start),
                    torch.ones(len(samples_collated[0]), 1)
                ), dim=1)
                samples_modified.append(current_sample)
            else:
                samples_modified.append(torch.cat((
                    samples_collated[0][:, start:end],
                    torch.zeros(len(samples_collated[0]), 1)
                ), dim=1))

                if noise_mask[:, index].any():
                    current_mask = noise_mask[:, index]
                    noise_feature = torch.rand(current_mask.sum())
                    noise_zero = noise_feature < mask_probability
                    noise_replace = mask_probability <= noise_feature
                    noise_replace.logical_and_(noise_feature < (mask_probability + replace_probability))
                    if noise_zero.any():
                        current_feature = samples_modified[-1][current_mask]
                        current_feature[noise_zero, :-1] = 0.0
                        current_feature[noise_zero, -1] = 1.0
                        samples_modified[-1][current_mask] = current_feature
                    if noise_replace.any():
                        current_feature = samples_modified[-1][current_mask]
                        if index in feature_discrete:
                            current_feature[noise_replace, :-1] = torch.nn.functional.one_hot(
                                input=torch.randint(low=0, high=end - start, size=(noise_replace.sum(),)),
                                num_classes=sizes[index]
                            ).to(dtype=samples_modified[-1].dtype)
                            samples_modified[-1][current_mask] = current_feature
                        else:
                            current_feature[noise_replace, :-1] = torch.randn(
                                noise_replace.sum(), end - start)
                            samples_modified[-1][current_mask] = current_feature

            # Mask out missing features.
            missing_mask = torch.as_tensor([]) if len(samples_collated) <= 1 else samples_collated[1][:, index]
            missing_mask_count = missing_mask.sum()
            if missing_mask_count > 0:
                missing_sample = torch.zeros(missing_mask_count, end - start + 1)
                samples_modified[-1][missing_mask] = missing_sample

        # Adapt noise mask to include targets.
        if len(target_discrete) > 0:
            noise_mask[:, target_discrete] = True
        if len(target_numeric) > 0:
            noise_mask[:, target_numeric] = True

        return torch.cat(samples_modified, dim=1), noise_mask, *samples_collated

    def _get_subset(self,
                    indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the specified subset from the dataset.

        :param indices: indices of the subset samples
        :return: specified subset
        """
        data, data_missing = [], []
        for index in indices:
            current_sample = self.dataset[index]
            data.append(current_sample[0])
            data_missing.append(current_sample[1])
        return torch.stack(data, dim=0), torch.stack(data_missing, dim=0)

    def setup(self,
              stage: Optional[str] = None) -> None:
        """
        Set up the specified stage of the data module.

        :param stage: stage to set up
        :return: None
        """
        if stage in (TrainerFn.FITTING, r'memory'):
            assert self.dataset.split_train is not None, r'No training samples specified!'
            data_train = self._get_subset(indices=self.dataset.split_train)
            self.__data_train = ConcatDataset([
                TensorDataset(*data_train, self.dataset.split_train) for _ in range(self.__super_sample_factor)
            ])
            if self.memory is None:
                self.memory = self.scale_and_noise_collate(
                    samples=list(zip(*data_train)),
                    mean=self.dataset.feature_mean,
                    stdv=self.dataset.feature_stdv,
                    sizes=self.dataset.sizes,
                    noise_probability=0.0,
                    mask_probability=0.0,
                    replace_probability=0.0,
                    target_discrete=self.dataset.target_discrete,
                    target_numeric=self.dataset.target_numeric,
                    feature_discrete=self.dataset.feature_discrete,
                    exclude_targets=False
                )[0]
        if stage in (TrainerFn.FITTING, TrainerFn.VALIDATING):
            assert self.dataset.split_validation is not None, r'No validation samples specified!'
            data_validation = self._get_subset(indices=self.dataset.split_validation)
            self.__data_validation = TensorDataset(*data_validation)
        elif stage == TrainerFn.TESTING:
            assert self.dataset.split_test is not None, r'No test samples specified!'
            data_test = self._get_subset(indices=self.dataset.split_test)
            self.__data_test = TensorDataset(*data_test)

    def train_dataloader(self) -> DataLoader:
        """
        Prepare and get the data loader for the training subset.

        :return: data loader for the training subset
        """
        return DataLoader(
            dataset=self.__data_train,
            batch_size=len(self.__data_train) if self.__batch_size is None else self.__batch_size,
            pin_memory=True,
            num_workers=self.__num_workers,
            persistent_workers=self.__num_workers > 0,
            collate_fn=partial(
                self.scale_and_noise_collate,
                mean=self.dataset.feature_mean,
                stdv=self.dataset.feature_stdv,
                sizes=self.dataset.sizes,
                noise_probability=self.__noise_probability,
                mask_probability=self.__mask_probability,
                replace_probability=self.__replace_probability,
                target_discrete=self.dataset.target_discrete,
                target_numeric=self.dataset.target_numeric,
                feature_discrete=self.dataset.feature_discrete,
                exclude_targets=True
            )
        )

    def val_dataloader(self) -> DataLoader:
        """
        Prepare and get the data loader for the validation subset.

        :return: data loader for the validation subset
        """
        return DataLoader(
            dataset=self.__data_validation,
            batch_size=len(self.__data_validation),
            pin_memory=True,
            num_workers=self.__num_workers,
            persistent_workers=self.__num_workers > 0,
            collate_fn=partial(
                self.scale_and_noise_collate,
                mean=self.dataset.feature_mean,
                stdv=self.dataset.feature_stdv,
                sizes=self.dataset.sizes,
                noise_probability=0.0,
                mask_probability=0.0,
                replace_probability=0.0,
                target_discrete=self.dataset.target_discrete,
                target_numeric=self.dataset.target_numeric,
                feature_discrete=self.dataset.feature_discrete,
                exclude_targets=True
            )
        )

    def test_dataloader(self) -> DataLoader:
        """
        Prepare and get the data loader for the test subset.

        :return: data loader for the test subset
        """
        return DataLoader(
            dataset=self.__data_test,
            batch_size=len(self.__data_test),
            pin_memory=True,
            num_workers=self.__num_workers,
            persistent_workers=self.__num_workers > 0,
            collate_fn=partial(
                self.scale_and_noise_collate,
                mean=self.dataset.feature_mean,
                stdv=self.dataset.feature_stdv,
                sizes=self.dataset.sizes,
                noise_probability=0.0,
                mask_probability=0.0,
                replace_probability=0.0,
                target_discrete=self.dataset.target_discrete,
                target_numeric=self.dataset.target_numeric,
                feature_discrete=self.dataset.feature_discrete,
                exclude_targets=True
            )
        )


class CSVDataset(BaseDataset, metaclass=ABCMeta):
    """
    Abstract base class of a dataset in CSV format to be used in Hopular.
    """

    def __init__(self,
                 dataset_name: str,
                 feature_numeric: Optional[Iterable[int]],
                 feature_discrete: Optional[Iterable[int]],
                 target_numeric: Optional[Iterable[int]],
                 target_discrete: Optional[Iterable[int]],
                 missing_entries: Optional[Dict[int, Iterable[int]]],
                 split_index: int,
                 num_splits: Optional[int],
                 split_state: Optional[int],
                 validation_size: Optional[float],
                 unique_only: bool,
                 checkpoint_mode: BaseDataset.CheckpointMode):
        """
        Partially initialize a CSV dataset to be used in Hopular.

        :param dataset_name: name of the dataset to be loaded
        :param feature_numeric: indices of continuous attributes (features and targets)
        :param feature_discrete: indices of discrete attributes (features and targets)
        :param target_numeric: indices of continuous targets
        :param target_discrete: indices of discrete targets
        :param missing_entries: positions of missing entries
        :param split_index: index of the split to be used
        :param num_splits: count of dataset splits
        :param split_state: random state indicative for a specific splitting
        :param validation_size: proportion of the validation subset
        :param unique_only: only include unique samples
        :param checkpoint_mode: checkpoint mode used for validating Hopular
        """
        super(CSVDataset, self).__init__()
        self.__dataset_name = dataset_name
        self.__target_numeric = np.asarray([] if target_numeric is None else target_numeric, dtype=np.int_)
        self.__target_discrete = np.asarray([] if target_discrete is None else target_discrete, dtype=np.int_)
        self.__feature_numeric = np.asarray([] if feature_numeric is None else np.union1d(
            np.array(feature_numeric).reshape(-1), np.array(self.__target_numeric).reshape(-1)), dtype=np.int_)
        self.__feature_discrete = np.asarray([] if feature_discrete is None else np.union1d(
            np.array(feature_discrete).reshape(-1), np.array(self.__target_discrete).reshape(-1)), dtype=np.int_)
        assert (len(self.__feature_numeric) + len(self.__feature_discrete)) >= 1, r'Invalid features specified!'
        assert (len(self.__target_numeric) + len(self.__target_discrete)) >= 1, r'Invalid targets specified!'
        assert len(np.intersect1d(self.__feature_numeric, self.__feature_discrete)) == 0, r'Invalid features specified!'
        assert all((_ in self.__feature_numeric for _ in self.__target_numeric)), r'Invalid targets specified!'
        assert all((_ in self.__feature_discrete for _ in self.__target_discrete)), r'Invalid targets specified!'
        assert len(np.intersect1d(self.__target_numeric, self.__target_discrete)) == 0, r'Invalid targets specified!'

        self.__split_index = split_index
        self.__num_splits = 1 if num_splits is None else num_splits
        self.__split_state = split_state
        self.__validation_size = validation_size
        self.__unique_only = unique_only
        self.__checkpoint_mode = checkpoint_mode

        ########################################
        self.__resources_path = pathlib.Path(os.path.join("datasets","resources",self.__dataset_name))
        ########################################

        self.__splits = None
        self.__data = None
        self.__data_missing = dict() if missing_entries is None else missing_entries
        self.__data_missing_inverse = dict()
        for sample_index, column_indices in self.__data_missing.items():
            for column_index in column_indices:
                self.__data_missing_inverse.setdefault(column_index, set()).add(sample_index)

        self.__sizes = None
        self.__data = None
        self.reset()

    def __len__(self) -> int:
        """
        Get sample count of the dataset, disregarding any subset boundaries.

        :return: size of the complete dataset
        """
        return len(self.__data)

    def __getitem__(self,
                    item_index: int) -> Tuple[torch.Tensor, ...]:
        """
        Get specified encoded sample.

        :param item_index: index of sample to be encoded
        :return: specified encoded sample
        """
        missing_features = torch.zeros(self.__data[item_index].shape, dtype=bool)
        if (len(self.__data_missing) > 0) and (item_index in self.__data_missing):
            missing_features[torch.as_tensor(self.__data_missing[item_index])] = True
        return self.encode_sample(sample=self.__data[item_index]), missing_features

    @abstractmethod
    def _load_data(self) -> np.ndarray:
        """
        Abstract method to be implemented by inheriting dataset classes.
        Load raw specified dataset without further pre-processing.

        :return: raw dataset
        """
        pass

    def reset(self) -> None:
        """
        Reset the current dataset instance by freshly pre-processing the stored raw dataset.

        :return: None
        """
        self.__data = self._load_data()

        # Map discrete non-numeric features to numeric values.
        self.__sizes = []
        for column_index in range(self.__data.shape[1]):
            column_data = self.__data[:, column_index].copy()
            valid_sample_indices = np.arange(len(column_data))
            if (len(self.__data_missing_inverse) > 0) and (column_index in self.__data_missing_inverse):
                missing_sample_indices = np.asarray(list(self.__data_missing_inverse[column_index]), dtype=np.int_)
                valid_sample_indices = np.setdiff1d(valid_sample_indices, missing_sample_indices)
                column_data[missing_sample_indices] = float(0)

            # Map columns to specified feature type.
            if column_index in self.__feature_numeric:
                self.__data[:, column_index] = column_data.astype(float)
                self.__sizes.append(1)
            elif column_index in self.__feature_discrete:
                column_data_unique = sorted(set(column_data[valid_sample_indices]))
                col_mapping = {v: float(k) for k, v in enumerate(column_data_unique)}
                column_data[valid_sample_indices] = np.vectorize(col_mapping.get)(column_data[valid_sample_indices])
                self.__data[:, column_index] = column_data
                self.__sizes.append(len(col_mapping))

        # Drop unspecified features and ensure floating point data type.
        specified_features = np.union1d(self.__feature_numeric, self.__feature_discrete)
        dropped_features = np.setdiff1d(np.arange(self.__data.shape[1]), specified_features).reshape(-1, 1)
        self.__data = self.__data[:, specified_features].astype(float)

        # Adapt feature/target indices.
        self.__feature_numeric -= (dropped_features < self.__feature_numeric.reshape(1, -1)).sum(axis=0)
        self.__feature_discrete -= (dropped_features < self.__feature_discrete.reshape(1, -1)).sum(axis=0)
        self.__target_numeric -= (dropped_features < self.__target_numeric.reshape(1, -1)).sum(axis=0)
        self.__target_discrete -= (dropped_features < self.__target_discrete.reshape(1, -1)).sum(axis=0)

        # Preprocess features and labels.
        unique_indices = np.arange(len(self.__data))
        if self.__unique_only:
            _, unique_indices = np.unique(self.__data, axis=0, return_index=True)
            unique_indices = np.sort(unique_indices)
            self.__data = self.__data[unique_indices]

        # Adapt missing data indices.
        if len(self.__data_missing) > 0:
            valid_indices = np.intersect1d(unique_indices, np.asarray(list(self.__data_missing.keys())))
            self.__data_missing = {
                index: (self.__data_missing[index] - (dropped_features < np.asarray(
                    self.__data_missing[index]).reshape(1, -1)).sum(axis=0)) for index in valid_indices
            }

        # Parse (fixed) split information.
        if self.__num_splits < 2:
            split_test = np.where(pd.read_csv(
                self.__resources_path / r'folds_py.dat',
                header=None, dtype=float, usecols=[self.__split_index]
            ).iloc[unique_indices] == 1)[0]
            split_validation = np.where(pd.read_csv(
                self.__resources_path / r'validation_folds_py.dat',
                header=None, dtype=float, usecols=[self.__split_index]
            ).iloc[unique_indices] == 1)[0]
            split_training = np.arange(self.__data.shape[0])[np.setdiff1d(
                np.arange(self.__data.shape[0]), np.concatenate((split_test, split_validation), axis=0))]
        else:
            assert self.__split_state is not None, r'Invalid splitting state!'
            assert self.__validation_size is not None and 0 < self.__validation_size < 1, r'Invalid validation size!'
            is_stratified = not any((
                len(self.__target_numeric) > 0,
                (len(self.__target_discrete) + len(self.__target_numeric)) > 1
            ))
            splitting = StratifiedKFold if is_stratified else KFold
            splitting = splitting(n_splits=self.__num_splits, shuffle=True, random_state=self.__split_state)

            # Create training, validation and test split – either randomly or stratified.
            splits_training_validation, split_test = list(splitting.split(
                np.arange(self.__data.shape[0]),
                self.__data[:, self.__target_discrete] if is_stratified else None
            ))[self.__split_index]
            split_training, split_validation = train_test_split(
                splits_training_validation,
                test_size=self.__validation_size,
                random_state=self.__split_state,
                shuffle=True,
                stratify=self.__data[splits_training_validation, self.__target_discrete] if is_stratified else None
            )

        # Sanity check overlapping indices.
        assert len(np.intersect1d(split_training, split_validation)) == 0, r'Overlapping training/validation sets!'
        assert len(np.intersect1d(split_training, split_test)) == 0, r'Overlapping training/test sets!'
        assert len(np.intersect1d(split_validation, split_test)) == 0, r'Overlapping validation/test sets!'
        self.__splits = (split_training, split_validation, split_test)

        # Sort dataset according to splits.
        self.__data = np.concatenate((
            self.__data[split_training], self.__data[split_validation], self.__data[split_test]
        ), axis=0)

        # Encapsulate data in corresponding container.
        self.__data = torch.as_tensor(self.__data, dtype=torch.float32)
        self.__feature_numeric = torch.as_tensor(self.__feature_numeric)
        self.__feature_discrete = torch.as_tensor(self.__feature_discrete)
        self.__target_numeric = torch.as_tensor(self.__target_numeric)
        self.__target_discrete = torch.as_tensor(self.__target_discrete)
        self.__splits = tuple(torch.as_tensor(split) for split in self.__splits)

    @property
    @lru_cache(maxsize=None)
    def feature_mean(self) -> Optional[torch.Tensor]:
        return torch.as_tensor([
            (self.__data[self.split_train, index].mean()) if index in self.feature_numeric else
            (float(r'nan')) for index in range(len(self.sizes))
        ])

    @property
    @lru_cache(maxsize=None)
    def feature_stdv(self) -> Optional[torch.Tensor]:
        return torch.as_tensor([
            (self.__data[self.split_train, index].std()) if index in self.feature_numeric else
            (float(r'nan')) for index in range(len(self.sizes))
        ])

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.__data.shape)

    @property
    def sizes(self) -> Tuple[int, ...]:
        return self.__sizes

    @property
    def split_train(self) -> torch.Tensor:
        return self.__splits[0].to(dtype=torch.long)
    @property
    def split_validation(self) -> torch.Tensor:
        return self.__splits[1].to(dtype=torch.long)

    @property
    def split_test(self) -> torch.Tensor:
        return self.__splits[2].to(dtype=torch.long)

    @property
    def feature_numeric(self) -> torch.Tensor:
        return self.__feature_numeric.to(dtype=torch.long)

    @property
    def feature_discrete(self) -> torch.Tensor:
        return self.__feature_discrete.to(dtype=torch.long)

    @property
    def target_numeric(self) -> torch.Tensor:
        return self.__target_numeric.to(dtype=torch.long)

    @property
    def target_discrete(self) -> torch.Tensor:
        return self.__target_discrete.to(dtype=torch.long)

    @property
    def dataset_name(self) -> str:
        return self.__dataset_name

    @property
    def resources_path(self) -> str:
        return self.__resources_path

    @property
    def split_index(self) -> int:
        return self.__split_index

    @property
    def num_splits(self) -> Optional[int]:
        return self.__num_splits

    @property
    def split_state(self) -> Optional[int]:
        return self.__split_state

    @property
    def validation_size(self) -> Optional[float]:
        return self.__validation_size

    @property
    def checkpoint_mode(self) -> BaseDataset.CheckpointMode:
        return self.__checkpoint_mode


class FixedDataset(CSVDataset):
    """
    Base class of a dataset in CSV format with fixed splits to be used in Hopular.
    """

    def __init__(self,
                 dataset_name: str,
                 feature_numeric: Iterable[int],
                 feature_discrete: Optional[Iterable[int]],
                 target_numeric: Iterable[int],
                 target_discrete: Iterable[int],
                 missing_entries: Optional[Dict[int, Iterable[int]]],
                 split_index: int,
                 split_state: int):
        """
        Initialize a CSV dataset with fixed splits to be used in Hopular.

        :param dataset_name: name of the dataset to be loaded
        :param feature_numeric: indices of continuous attributes (features and targets)
        :param feature_discrete: indices of discrete attributes (features and targets)
        :param target_numeric: indices of continuous targets
        :param target_discrete: indices of discrete targets
        :param missing_entries: positions of missing entries
        :param split_index: index of the split to be used
        :param split_state: random state indicative for a specific splitting
        """
        super(FixedDataset, self).__init__(
            dataset_name=dataset_name,
            feature_numeric=feature_numeric,
            feature_discrete=feature_discrete,
            target_numeric=target_numeric,
            target_discrete=target_discrete,
            missing_entries=missing_entries,
            split_index=split_index,
            num_splits=None,
            split_state=split_state,
            validation_size=None,
            unique_only=False,
            checkpoint_mode=BaseDataset.CheckpointMode.MAX
        )

    def _load_data(self) -> np.ndarray:
        features = pd.read_csv(self.resources_path / f'{self.dataset_name}_py.dat', header=None)
        labels = pd.read_csv(self.resources_path / r'labels_py.dat', header=None)
        return np.concatenate((features, labels), axis=1)


class CVDataset(CSVDataset):
    """
    Base class of a dataset in CSV format with random splits to be used in Hopular.
    """

    def __init__(self,
                 dataset_name: str,
                 feature_numeric: Iterable[int],
                 feature_discrete: Optional[Iterable[int]],
                 target_numeric: Iterable[int],
                 target_discrete: Iterable[int],
                 missing_entries: Optional[Dict[int, Iterable[int]]],
                 skip_rows: int,
                 split_index: int,
                 num_splits: int,
                 split_state: int,
                 validation_size: float,
                 checkpoint_mode: BaseDataset.CheckpointMode):
        """
        Initialize a CSV dataset with random splits to be used in Hopular.

        :param dataset_name: name of the dataset to be loaded
        :param feature_numeric: indices of continuous attributes (features and targets)
        :param feature_discrete: indices of discrete attributes (features and targets)
        :param target_numeric: indices of continuous targets
        :param target_discrete: indices of discrete targets
        :param missing_entries: positions of missing entries
        :param skip_rows: count of rows to be skipped when loading dataset from file
        :param split_index: index of the split to be used
        :param num_splits: count of dataset splits
        :param split_state: random state indicative for a specific splitting
        :param validation_size: proportion of the validation subset
        :param checkpoint_mode: checkpoint mode used for validating Hopular
        """
        self.__skip_rows = skip_rows
        self.__num_splits = num_splits
        super(CVDataset, self).__init__(
            dataset_name=dataset_name,
            feature_numeric=feature_numeric,
            feature_discrete=feature_discrete,
            target_numeric=target_numeric,
            target_discrete=target_discrete,
            missing_entries=missing_entries,
            split_index=split_index,
            num_splits=num_splits,
            split_state=split_state,
            validation_size=validation_size,
            unique_only=True,
            checkpoint_mode=checkpoint_mode
        )

    def _load_data(self) -> np.ndarray:
        return pd.read_csv(
            self.resources_path / f'{self.dataset_name}.csv',
            header=None,
            skiprows=self.__skip_rows
        ).to_numpy()


#######################################
class DiabetesDataset(FixedDataset):
    """
    Implementation of the diabetes dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <diabetes>.

        :param split_index: index of the split to be used
        """

        with open(r"datasets\resources\diabetes\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(DiabetesDataset, self).__init__(
            dataset_name=r'diabetes',
            feature_numeric=np.concatenate((np.asarray([7]),
                                            np.asarray([10]),
                                            np.asarray([11]),
                                            np.asarray([12]),
                                            np.asarray([13]),
                                            np.asarray([14]),
                                            np.asarray([15]),
                                            np.asarray([19]))),
            feature_discrete=np.concatenate((np.asarray([0]),
                                            np.asarray([1]),
                                            np.asarray([2]),
                                            np.asarray([3]),
                                            np.asarray([4]),
                                            np.asarray([5]),
                                            np.asarray([6]),
                                            np.asarray([8]),
                                            np.asarray([9]),
                                            np.asarray([16]),
                                            np.asarray([17]),
                                            np.asarray([18]),
                                            np.asarray([20]),
                                            np.asarray([21]),
                                            np.asarray([22]),
                                            np.asarray([23]),
                                            np.asarray([24]),
                                            np.asarray([25]),
                                            np.asarray([26]),
                                            np.asarray([27]),
                                            np.asarray([28]),
                                            np.asarray([29]),
                                            np.asarray([30]),
                                            np.asarray([31]),
                                            np.asarray([32]),
                                            np.asarray([33]),
                                            np.asarray([34]),
                                            np.asarray([35]),
                                            np.asarray([36]),
                                            np.asarray([37]),
                                            np.asarray([38]),
                                             np.asarray([39]))),
            target_numeric=None,
            target_discrete=[39],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class DiabetesSmallDataset(FixedDataset):
    """
    Implementation of the diabetessmall dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <diabetessmall>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\diabetessmall\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(DiabetesSmallDataset, self).__init__(
            dataset_name=r'diabetessmall',
            feature_numeric=np.concatenate((np.asarray([7]),
                                            np.asarray([10]),
                                            np.asarray([11]),
                                            np.asarray([13]),
                                            np.asarray([14]),
                                            np.asarray([15]),
                                            np.asarray([19]))),
            feature_discrete=np.concatenate((np.asarray([0]),
                                            np.asarray([1]),
                                            np.asarray([3]),
                                            np.asarray([4]),
                                            np.asarray([5]),
                                            np.asarray([6]),
                                            np.asarray([8]),
                                            np.asarray([9]),
                                            np.asarray([18]),
                                            np.asarray([20]),
                                            np.asarray([21]),
                                            np.asarray([24]),
                                            np.asarray([35]),
                                            np.asarray([37]),
                                            np.asarray([38]),
                                             np.asarray([39]))),
            target_numeric=None,
            target_discrete=[39],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class DiabetesNA60Dataset(FixedDataset):
    """
    Implementation of the diabetesNA60 dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <diabetesNA60>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\diabetesNA60\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(DiabetesNA60Dataset, self).__init__(
            dataset_name=r'diabetesNA60',
            feature_numeric=np.concatenate((np.asarray([7]),
                                            np.asarray([10]),
                                            np.asarray([11]),
                                            np.asarray([12]),
                                            np.asarray([13]),
                                            np.asarray([14]),
                                            np.asarray([15]),
                                            np.asarray([19]))),
            feature_discrete=np.concatenate((np.asarray([0]),
                                            np.asarray([1]),
                                            np.asarray([2]),
                                            np.asarray([3]),
                                            np.asarray([4]),
                                            np.asarray([5]),
                                            np.asarray([6]),
                                            np.asarray([8]),
                                            np.asarray([9]),
                                            np.asarray([16]),
                                            np.asarray([17]),
                                            np.asarray([18]),
                                            np.asarray([20]),
                                            np.asarray([21]),
                                            np.asarray([22]),
                                            np.asarray([23]),
                                            np.asarray([24]),
                                            np.asarray([25]),
                                            np.asarray([26]),
                                            np.asarray([27]),
                                            np.asarray([28]),
                                            np.asarray([29]),
                                            np.asarray([30]),
                                            np.asarray([31]),
                                            np.asarray([32]),
                                            np.asarray([33]),
                                            np.asarray([34]),
                                            np.asarray([35]),
                                            np.asarray([36]),
                                            np.asarray([37]),
                                            np.asarray([38]),
                                             np.asarray([39]))),
            target_numeric=None,
            target_discrete=[39],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class DiabetesNA60SmallDataset(FixedDataset):
    """
    Implementation of the diabetesNA60small dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <diabetesNA60small>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\diabetesNA60small\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(DiabetesNA60SmallDataset, self).__init__(
            dataset_name=r'diabetesNA60small',
            feature_numeric=np.concatenate((np.asarray([15]),
                                            np.asarray([19]))),
            feature_discrete=np.concatenate((np.asarray([1]),
                                            np.asarray([4]),
                                            np.asarray([5]),
                                            np.asarray([18]),
                                            np.asarray([21]),
                                            np.asarray([31]),
                                            np.asarray([32]),
                                            np.asarray([36]),
                                            np.asarray([37]),
                                            np.asarray([38]),
                                             np.asarray([39]))),
            target_numeric=None,
            target_discrete=[39],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class ThyroidDataset(FixedDataset):
    """
    Implementation of the thyroid dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <thyroid>.

        :param split_index: index of the split to be used
        """
        super(ThyroidDataset, self).__init__(
            dataset_name=r'thyroid',
            feature_numeric=np.concatenate((np.asarray([0]), np.arange(15,19))),
            feature_discrete=np.concatenate((np.arange(1,15),np.asarray([19]))),
            target_numeric=None,
            target_discrete=[19],
            missing_entries=None,
            split_index=split_index,
            split_state=1
        )


class ThyroidSmallDataset(FixedDataset):
    """
    Implementation of the thyroidsmall dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <thyroidsmall>.

        :param split_index: index of the split to be used
        """
        super(ThyroidSmallDataset, self).__init__(
            dataset_name=r'thyroidsmall',
            feature_numeric=np.concatenate((np.asarray([0]), np.arange(15,19))),
            feature_discrete=np.concatenate((np.arange(1,4),np.asarray([6]),np.asarray([7]),np.asarray([12]),np.asarray([13]),np.asarray([19]))),
            target_numeric=None,
            target_discrete=[19],
            missing_entries=None,
            split_index=split_index,
            split_state=1
        )


class ThyroidNA60Dataset(FixedDataset):
    """
    Implementation of the thyroidNA60 dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <thyroidNA60>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\thyroidNA60\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(ThyroidNA60Dataset, self).__init__(
            dataset_name=r'thyroidNA60',
            feature_numeric=np.concatenate((np.asarray([0]), np.arange(15,19))),
            feature_discrete=np.concatenate((np.arange(1,15),np.asarray([19]))),
            target_numeric=None,
            target_discrete=[19],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class ThyroidNA60SmallDataset(FixedDataset):
    """
    Implementation of the thyroidNA60small dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <thyroidNA60small>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\thyroidNA60small\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(ThyroidNA60SmallDataset, self).__init__(
            dataset_name=r'thyroidNA60small',
            feature_numeric=np.concatenate((np.asarray([0]), np.asarray([15]),np.asarray([17]),np.asarray([18]))),
            feature_discrete=np.concatenate((np.arange(1,3),np.asarray([6]),np.asarray([9]),np.asarray([19]))),
            target_numeric=None,
            target_discrete=[19],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class ThyroidNA20Dataset(FixedDataset):
    """
    Implementation of the thyroidNA20 dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <thyroidNA20>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\thyroidNA20\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(ThyroidNA20Dataset, self).__init__(
            dataset_name=r'thyroidNA20',
            feature_numeric=np.concatenate((np.asarray([0]), np.arange(15,19))),
            feature_discrete=np.concatenate((np.arange(1,15),np.asarray([19]))),
            target_numeric=None,
            target_discrete=[19],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class ThyroidNA20SmallDataset(FixedDataset):
    """
    Implementation of the thyroidNA20small dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <thyroidNA20small>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\thyroidNA20small\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(ThyroidNA20SmallDataset, self).__init__(
            dataset_name=r'thyroidNA20small',
            feature_numeric=np.concatenate((np.asarray([0]), np.arange(15,19))),
            feature_discrete=np.concatenate((np.arange(1,3),np.asarray([8]),np.asarray([9]),np.asarray([19]))),
            target_numeric=None,
            target_discrete=[19],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class ThyroidNA20NA60Dataset(FixedDataset):
    """
    Implementation of the thyroidNA20NA60 dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <thyroidNA20NA60>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\thyroidNA20NA60\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(ThyroidNA20NA60Dataset, self).__init__(
            dataset_name=r'thyroidNA20NA60',
            feature_numeric=np.concatenate((np.asarray([0]), np.arange(15,19))),
            feature_discrete=np.concatenate((np.arange(1,15),np.asarray([19]))),
            target_numeric=None,
            target_discrete=[19],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class ThyroidNA20NA60SmallDataset(FixedDataset):
    """
    Implementation of the thyroidNA20NA60small dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <thyroidNA20NA60small>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\thyroidNA20NA60small\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(ThyroidNA20NA60SmallDataset, self).__init__(
            dataset_name=r'thyroidNA20NA60small',
            feature_numeric=np.concatenate((np.asarray([0]), np.arange(15,19))),
            feature_discrete=np.concatenate((np.arange(1,3),np.arange(4,7),np.arange(8,10),np.asarray([19]))),
            target_numeric=None,
            target_discrete=[19],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class MyocardialDataset(FixedDataset):
    """
    Implementation of the myocardial dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <myocardial>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\myocardial\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(MyocardialDataset, self).__init__(
            dataset_name=r'myocardial',
            feature_numeric=np.arange(0,12),
            feature_discrete=np.arange(12,74),
            target_numeric=None,
            target_discrete=[73],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class MyocardialSmallDataset(FixedDataset):
    """
    Implementation of the myocardialsmall dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <myocardialsmall>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\myocardialsmall\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(MyocardialSmallDataset, self).__init__(
            dataset_name=r'myocardialsmall',
            feature_numeric=np.concatenate((np.asarray([0]),
                                            np.asarray([1]),
                                            np.asarray([3]),
                                            np.asarray([4]),
                                            np.asarray([5]),
                                            np.asarray([6]),
                                            np.asarray([7]),
                                            np.asarray([8]),
                                            np.asarray([10]),
                                            np.asarray([11]))),
            feature_discrete=np.concatenate((np.asarray([14]),
                                            np.asarray([16]),
                                            np.asarray([21]),
                                            np.asarray([25]),
                                            np.asarray([26]),
                                            np.asarray([27]),
                                            np.asarray([29]),
                                            np.asarray([33]),
                                            np.asarray([34]),
                                            np.asarray([38]),
                                            np.asarray([39]),
                                            np.asarray([40]),
                                            np.asarray([41]),
                                            np.asarray([42]),
                                            np.asarray([43]),
                                            np.asarray([44]),
                                            np.asarray([45]),
                                            np.asarray([46]),
                                            np.asarray([51]),
                                            np.asarray([52]),
                                            np.asarray([53]),
                                            np.asarray([56]),
                                            np.asarray([61]),
                                            np.asarray([62]),
                                            np.asarray([66]),
                                             np.asarray([73]))),
            target_numeric=None,
            target_discrete=[73],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class MyocardialRecDataset(FixedDataset):
    """
    Implementation of the myocardialrec dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <myocardialrec>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\myocardialrec\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(MyocardialRecDataset, self).__init__(
            dataset_name=r'myocardialrec',
            feature_numeric=np.arange(0,12),
            feature_discrete=np.arange(12,74),
            target_numeric=None,
            target_discrete=[73],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class MyocardialRecSmallDataset(FixedDataset):
    """
    Implementation of the myocardialrecsmall dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <myocardialrecsmall>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\myocardialrecsmall\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(MyocardialRecSmallDataset, self).__init__(
            dataset_name=r'myocardialrecsmall',
            feature_numeric=np.concatenate((np.asarray([1]),
                                            np.asarray([6]),
                                            np.asarray([8]),
                                            np.asarray([10]))),
            feature_discrete=np.concatenate((np.asarray([12]),
                                            np.asarray([14]),
                                            np.asarray([16]),
                                            np.asarray([17]),
                                            np.asarray([26]),
                                            np.asarray([36]),
                                            np.asarray([41]),
                                            np.asarray([42]),
                                            np.asarray([43]),
                                            np.asarray([47]),
                                            np.asarray([48]),
                                            np.asarray([50]),
                                            np.asarray([53]),
                                            np.asarray([61]),
                                            np.asarray([62]),
                                            np.asarray([64]),
                                            np.asarray([66]),
                                            np.asarray([67]),
                                            np.asarray([69]),
                                            np.asarray([70]),
                                             np.asarray([73]))),
            target_numeric=None,
            target_discrete=[73],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class MyocardialNA60Dataset(FixedDataset):
    """
    Implementation of the myocardialNA60 dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <myocardialNA60>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\myocardialNA60\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(MyocardialNA60Dataset, self).__init__(
            dataset_name=r'myocardialNA60',
            feature_numeric=np.arange(0,12),
            feature_discrete=np.arange(12,74),
            target_numeric=None,
            target_discrete=[73],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class MyocardialNA60SmallDataset(FixedDataset):
    """
    Implementation of the myocardialNA60small dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <myocardialNA60small>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\myocardialNA60small\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(MyocardialNA60SmallDataset, self).__init__(
            dataset_name=r'myocardialNA60small',
            feature_numeric=np.concatenate((np.asarray([0]),
                                            np.asarray([1]),
                                            np.asarray([3]),
                                            np.asarray([8]))),
            feature_discrete=np.concatenate((np.asarray([12]),
                                            np.asarray([13]),
                                            np.asarray([14]),
                                            np.asarray([16]),
                                            np.asarray([19]),
                                            np.asarray([21]),
                                            np.asarray([22]),
                                            np.asarray([23]),
                                            np.asarray([24]),
                                            np.asarray([25]),
                                            np.asarray([26]),
                                            np.asarray([29]),
                                            np.asarray([32]),
                                            np.asarray([33]),
                                            np.asarray([37]),
                                            np.asarray([38]),
                                            np.asarray([39]),
                                            np.asarray([41]),
                                            np.asarray([42]),
                                            np.asarray([43]),
                                            np.asarray([44]),
                                            np.asarray([49]),
                                            np.asarray([52]),
                                            np.asarray([56]),
                                            np.asarray([62]),
                                            np.asarray([64]),
                                            np.asarray([68]),
                                            np.asarray([69]),
                                             np.asarray([73]))),
            target_numeric=None,
            target_discrete=[73],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class MyocardialRecNA60Dataset(FixedDataset):
    """
    Implementation of the myocardialrecNA60 dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <myocardialrecNA60>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\myocardialrecNA60\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(MyocardialRecNA60Dataset, self).__init__(
            dataset_name=r'myocardialrecNA60',
            feature_numeric=np.arange(0,12),
            feature_discrete=np.arange(12,74),
            target_numeric=None,
            target_discrete=[73],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class MyocardialRecNA60SmallDataset(FixedDataset):
    """
    Implementation of the myocardialrecNA60small dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <myocardialrecNA60small>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\myocardialrecNA60small\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(MyocardialRecNA60SmallDataset, self).__init__(
            dataset_name=r'myocardialrecNA60small',
            feature_numeric=np.concatenate((np.asarray([0]),
                                            np.asarray([3]),
                                            np.asarray([10]))),
            feature_discrete=np.concatenate((np.asarray([12]),
                                            np.asarray([16]),
                                            np.asarray([19]),
                                            np.asarray([21]),
                                            np.asarray([24]),
                                            np.asarray([26]),
                                            np.asarray([28]),
                                            np.asarray([29]),
                                            np.asarray([38]),
                                            np.asarray([51]),
                                            np.asarray([52]),
                                            np.asarray([53]),
                                            np.asarray([55]),
                                            np.asarray([56]),
                                            np.asarray([61]),
                                            np.asarray([62]),
                                            np.asarray([64]),
                                            np.asarray([65]),
                                            np.asarray([66]),
                                            np.asarray([67]),
                                            np.asarray([71]),
                                             np.asarray([73]))),
            target_numeric=None,
            target_discrete=[73],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class ThyroidNA30Dataset(FixedDataset):
    """
    Implementation of the thyroidNA30 dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <thyroidNA30>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\thyroidNA30\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(ThyroidNA30Dataset, self).__init__(
            dataset_name=r'thyroidNA30',
            feature_numeric=np.concatenate((np.asarray([0]),
                                            np.asarray([15]),
                                            np.asarray([16]),
                                            np.asarray([17]),
                                            np.asarray([18]))),
            feature_discrete=np.concatenate((np.asarray([1]),
                                            np.asarray([2]),
                                            np.asarray([3]),
                                            np.asarray([4]),
                                            np.asarray([5]),
                                            np.asarray([6]),
                                            np.asarray([7]),
                                            np.asarray([8]),
                                            np.asarray([9]),
                                            np.asarray([10]),
                                            np.asarray([11]),
                                            np.asarray([12]),
                                            np.asarray([13]),
                                            np.asarray([14]),
                                            np.asarray([19]))),
            target_numeric=None,
            target_discrete=[19],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class ThyroidNA30SmallDataset(FixedDataset):
    """
    Implementation of the thyroidNA30small dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <thyroidNA30small>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\thyroidNA30small\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(ThyroidNA30SmallDataset, self).__init__(
            dataset_name=r'thyroidNA30small',
            feature_numeric=np.concatenate((np.asarray([0]),
                                            np.asarray([15]),
                                            np.asarray([16]),
                                            np.asarray([17]),
                                            np.asarray([18]))),
            feature_discrete=np.concatenate((np.asarray([1]),
                                            np.asarray([2]),
                                            np.asarray([8]),
                                            np.asarray([9]),
                                            np.asarray([10]),
                                            np.asarray([19]))),
            target_numeric=None,
            target_discrete=[19],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class ThyroidNA30NA60Dataset(FixedDataset):
    """
    Implementation of the thyroidNA30NA60 dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <thyroidNA30NA60>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\thyroidNA30NA60\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(ThyroidNA30NA60Dataset, self).__init__(
            dataset_name=r'thyroidNA30NA60',
            feature_numeric=np.concatenate((np.asarray([0]),
                                            np.asarray([15]),
                                            np.asarray([16]),
                                            np.asarray([17]),
                                            np.asarray([18]))),
            feature_discrete=np.concatenate((np.asarray([1]),
                                             np.asarray([2]),
                                             np.asarray([3]),
                                             np.asarray([4]),
                                             np.asarray([5]),
                                             np.asarray([6]),
                                             np.asarray([7]),
                                             np.asarray([8]),
                                             np.asarray([9]),
                                             np.asarray([10]),
                                             np.asarray([11]),
                                             np.asarray([12]),
                                             np.asarray([13]),
                                             np.asarray([14]),
                                             np.asarray([19]))),
            target_numeric=None,
            target_discrete=[19],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class ThyroidNA30NA60SmallDataset(FixedDataset):
    """
    Implementation of the thyroidNA30NA60small dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <thyroidNA30NA60small>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\thyroidNA30NA60small\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(ThyroidNA30NA60SmallDataset, self).__init__(
            dataset_name=r'thyroidNA30NA60small',
            feature_numeric=np.concatenate((np.asarray([15]),
                                            np.asarray([16]),
                                            np.asarray([17]),
                                            np.asarray([18]))),
            feature_discrete=np.concatenate((np.asarray([1]),
                                            np.asarray([2]),
                                            np.asarray([5]),
                                            np.asarray([7]),
                                            np.asarray([19]))),
            target_numeric=None,
            target_discrete=[19],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class ThyroidNA10Dataset(FixedDataset):
    """
    Implementation of the thyroidNA10 dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <thyroidNA10>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\thyroidNA10\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(ThyroidNA10Dataset, self).__init__(
            dataset_name=r'thyroidNA10',
            feature_numeric=np.concatenate((np.asarray([0]),
                                            np.asarray([15]),
                                            np.asarray([16]),
                                            np.asarray([17]),
                                            np.asarray([18]))),
            feature_discrete=np.concatenate((np.asarray([1]),
                                            np.asarray([2]),
                                            np.asarray([3]),
                                            np.asarray([4]),
                                            np.asarray([5]),
                                            np.asarray([6]),
                                            np.asarray([7]),
                                            np.asarray([8]),
                                            np.asarray([9]),
                                            np.asarray([10]),
                                            np.asarray([11]),
                                            np.asarray([12]),
                                            np.asarray([13]),
                                            np.asarray([14]),
                                            np.asarray([19]))),
            target_numeric=None,
            target_discrete=[19],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class ThyroidNA10SmallDataset(FixedDataset):
    """
    Implementation of the thyroidNA10small dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <thyroidNA10small>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\thyroidNA10small\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(ThyroidNA10SmallDataset, self).__init__(
            dataset_name=r'thyroidNA10small',
            feature_numeric=np.concatenate((np.asarray([0]),
                                            np.asarray([15]),
                                            np.asarray([16]),
                                            np.asarray([17]),
                                            np.asarray([18]))),
            feature_discrete=np.concatenate((np.asarray([1]),
                                            np.asarray([2]),
                                            np.asarray([4]),
                                            np.asarray([6]),
                                            np.asarray([7]),
                                            np.asarray([11]),
                                            np.asarray([19]))),
            target_numeric=None,
            target_discrete=[19],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class ThyroidNA10NA60Dataset(FixedDataset):
    """
    Implementation of the thyroidNA10NA60 dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <thyroidNA10NA60>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\thyroidNA10NA60\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(ThyroidNA10NA60Dataset, self).__init__(
            dataset_name=r'thyroidNA10NA60',
            feature_numeric=np.concatenate((np.asarray([0]),
                                            np.asarray([15]),
                                            np.asarray([16]),
                                            np.asarray([17]),
                                            np.asarray([18]))),
            feature_discrete=np.concatenate((np.asarray([1]),
                                             np.asarray([2]),
                                             np.asarray([3]),
                                             np.asarray([4]),
                                             np.asarray([5]),
                                             np.asarray([6]),
                                             np.asarray([7]),
                                             np.asarray([8]),
                                             np.asarray([9]),
                                             np.asarray([10]),
                                             np.asarray([11]),
                                             np.asarray([12]),
                                             np.asarray([13]),
                                             np.asarray([14]),
                                             np.asarray([19]))),
            target_numeric=None,
            target_discrete=[19],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )


class ThyroidNA10NA60SmallDataset(FixedDataset):
    """
    Implementation of the thyroidNA10NA60small dataset.
    """

    def __init__(self,
                 split_index: int = 0):
        """
        Initialize dataset <thyroidNA10NA60small>.

        :param split_index: index of the split to be used
        """
        with open(r"datasets\resources\thyroidNA10NA60small\missing.json","r") as fp:
            missing_entries = json.load(fp)
        super(ThyroidNA10NA60SmallDataset, self).__init__(
            dataset_name=r'thyroidNA10NA60small',
            feature_numeric=np.concatenate((np.asarray([0]),
                                            np.asarray([15]),
                                            np.asarray([16]),
                                            np.asarray([17]),
                                            np.asarray([18]))),
            feature_discrete=np.concatenate((np.asarray([1]),
                                            np.asarray([3]),
                                            np.asarray([4]),
                                            np.asarray([9]),
                                            np.asarray([12]),
                                            np.asarray([13]),
                                            np.asarray([19]))),
            target_numeric=None,
            target_discrete=[19],
            missing_entries=missing_entries,
            split_index=split_index,
            split_state=1
        )
###########################


def list_datasets() -> List[str]:
    """
    Get the list of implemented datasets.

    :return: list of implemented datasets
    """
    return [dataset[0] for dataset in inspect.getmembers(
        object=sys.modules[__name__],
        predicate=lambda entity: (
                inspect.isclass(entity) and
                issubclass(entity, (FixedDataset, CVDataset)) and
                any(entity is not subtype for subtype in (FixedDataset, CVDataset))
        )
    )]


def find_dataset(name: str) -> ABCMeta:
    """
    Get specified dataset using a substring matching procedure.

    :param name: substring of dataset name
    :return: specified dataset if found
    """
    possible_datasets = inspect.getmembers(
        object=sys.modules[__name__],
        predicate=lambda entity: inspect.isclass(entity) and name.lower() in entity.__name__.lower()
    )
    assert len(possible_datasets) > 0, r'No dataset with query <{name}> found!'
    assert len(possible_datasets) == 1, r'Ambiguous dataset query <{name}>!'
    return possible_datasets[0][1]

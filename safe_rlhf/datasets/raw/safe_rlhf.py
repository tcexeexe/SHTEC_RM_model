#******************************************************************************************************** 
#  @author	     tcexeexe
#  @date         04,2024
#
#  @par     Copyright (c) 2024, SHTEC
# 
# *******************************************************************************************************/
"""Safe-RLHF preference datasets."""

from __future__ import annotations

from typing import ClassVar

from datasets import load_dataset
from safe_rlhf.datasets.base import RawDataset, RawSample


__all__ = [
    'SafeRLHFDataset',
    'SafeRLHFTrainDataset',
    'SafeRLHFTestDataset',
    'SafeRLHF30KTrainDataset',
    'SafeRLHF30KTestDataset',
    'SafeRLHF10KTrainDataset',
]


class SafeRLHFDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or self.PATH, split=self.SPLIT)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['prompt'],
            answer=data['response_0'],
            other_answer=data['response_1'],
            better=int(data['better_response_id']) == 0,
            safer=int(data['safer_response_id']) == 0,
            is_safe=bool(data['is_response_0_safe']),
            is_other_safe=bool(data['is_response_1_safe']),
        )

    def __len__(self) -> int:
        return len(self.data)


class SafeRLHFTrainDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF/train',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF'
    SPLIT: str = 'train'


class SafeRLHFTestDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF/test'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF/test',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF'
    SPLIT: str = 'test'


class SafeRLHF30KTrainDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF-30K/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-30K/train',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF-30K'
    SPLIT: str = 'train'


class SafeRLHF30KTestDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF-30K/test'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-30K/test',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF-30K'
    SPLIT: str = 'test'


class SafeRLHF10KTrainDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF-10K/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-10K/train',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF-10K'
    SPLIT: str = 'train'

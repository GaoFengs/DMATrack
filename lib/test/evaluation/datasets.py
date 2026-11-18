from collections import namedtuple
import importlib

from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple("DatasetInfo", ["module", "class_name", "kwargs"])

dataset_dict = {
    "lasher": DatasetInfo(
        module="lib.test.evaluation.lasherdataset",
        class_name="LasHeRDataset",
        kwargs=dict(),
    ),
}

def load_dataset(name: str):
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError("Unknown dataset '{}'".format(name))

    module = importlib.import_module(dset_info.module)
    dataset = getattr(module, dset_info.class_name)(**dset_info.kwargs)
    return dataset.get_sequence_list()

def get_dataset(*args, **kwargs):
    del kwargs
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset

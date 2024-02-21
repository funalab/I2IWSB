from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


def get_train_iterator(args, train_dataset, validation_dataset, generator):
    if hasattr(args, 'num_workers'):
        num_workers = int(args.num_workers)
    else:
        num_workers = 0
    if hasattr(args, 'pin_memory'):
        pin_memory = eval(args.pin_memory)
    else:
        pin_memory = False

    train_iterator = DataLoader(
        dataset=train_dataset,
        batch_size=int(args.batchsize),
        shuffle=True,
        generator=generator,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    validation_iterator = DataLoader(
        dataset=validation_dataset,
        batch_size=int(args.val_batchsize),
        shuffle=False,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_iterator, validation_iterator


def get_test_iterator(args, test_dataset):
    if hasattr(args, 'num_workers'):
        num_workers = int(args.num_workers)
    else:
        num_workers = 0
    if hasattr(args, 'pin_memory'):
        pin_memory = eval(args.pin_memory)
    else:
        pin_memory = False

    test_iterator = DataLoader(
        dataset=test_dataset,
        batch_size=int(args.val_batchsize),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return test_iterator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def setup_loader(dataset, batch_size, generator, num_workers=4):
    loader = DataLoaderX(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        generator=generator,
        num_workers=num_workers,
        drop_last=True,
    )

    return loader

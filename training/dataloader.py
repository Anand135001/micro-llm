from torch.utils.data import DataLoader

from data.stream_lm_dataset import StreamingLMDataset


def create_dataloader(
    batch_size: int = 2,
    validation: bool = False,
) -> DataLoader:

    dataset = StreamingLMDataset(
        validation=validation,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )
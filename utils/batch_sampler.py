import random
from torch.utils.data import Sampler

from .SID_Set import FakePartsV2Dataset


class BalancedBatchSampler(Sampler):
    """
    Simple class-balanced batch sampler compatible with Distributed. Mirrors utils.batch_sampler.BatchSampler.
    """
    def __init__(self, dataset: FakePartsV2Dataset, batch_size: int, world_size: int = 1, rank: int = 0):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank

        self.indices_by_class = {0: [], 1: []}
        for idx in range(len(dataset)):
            self.indices_by_class[int(dataset.cls_labels[idx])].append(idx)

        # Keep originals for reset
        self.original_indices = {cls: idxs.copy() for cls, idxs in self.indices_by_class.items()}

    def __iter__(self):
        self.indices_by_class = {cls: idxs.copy() for cls, idxs in self.original_indices.items()}
        all_batches = []
        for cls, indices in self.indices_by_class.items():
            indices = list(indices)
            import random as _r
            _r.shuffle(indices)
            for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
                all_batches.append(indices[i:i+self.batch_size])
        import random as _r
        _r.shuffle(all_batches)

        if self.world_size > 1:
            num_batches = len(all_batches)
            per_rank = num_batches // self.world_size
            start = self.rank * per_rank
            end = start + per_rank
            all_batches = all_batches[start:end]
        return iter(all_batches)

    def __len__(self):
        total = sum(len(idxs) // self.batch_size for idxs in self.indices_by_class.values())
        if self.world_size > 1:
            return total // self.world_size
        return total

class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size, world_size=1, rank=0):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        
        # Group indices by class
        self.indices_by_class = {
            0: [],  # real
            1: [],  # fake
        }
        
        # Populate indices_by_class
        for idx in range(len(dataset)):
            cls = dataset.cls_labels[idx]
            self.indices_by_class[cls].append(idx)
            
        print(f"\nRank {self.rank} - Dataset statistics:")
        for cls in self.indices_by_class:
            print(f"Class {cls}: {len(self.indices_by_class[cls])} images")
        
        # Store original indices
        self.original_indices = {cls: indices.copy() for cls, indices in self.indices_by_class.items()}
    
    def __iter__(self):
        # Reset indices at the start of each iteration
        self.indices_by_class = {cls: indices.copy() for cls, indices in self.original_indices.items()}
        
        # Create all possible batches for each class
        all_batches = []
        for cls in self.indices_by_class:
            indices = list(self.indices_by_class[cls])
            random.shuffle(indices)
            
            # Create complete batches for this class
            for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
                batch = indices[i:i + self.batch_size]
                all_batches.append(batch)
        
        # Shuffle batches to ensure randomness in class distribution between batches
        random.shuffle(all_batches)
        
        # Handle distributed training
        if self.world_size > 1:
            num_batches = len(all_batches)
            num_batches_per_rank = num_batches // self.world_size
            all_batches = all_batches[self.rank * num_batches_per_rank : (self.rank + 1) * num_batches_per_rank]
        
        return iter(all_batches)

    def __len__(self):
        total_batches = sum(len(indices) // self.batch_size for indices in self.indices_by_class.values())
        if self.world_size > 1:
            return total_batches // self.world_size
        return total_batches

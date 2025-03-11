import random
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from dataclasses import dataclass
from heapq import heappop, heappush

MIN_P_WEIGHT = -1.0

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "valid_actions", "reward")
)


class Buffer(ABC):
    def __init__(self, capacity):
        self.capacity = capacity

    @abstractmethod
    def push(self, *args, p_weight: float):
        pass

    @abstractmethod
    def sample(self, batch_size) -> tuple[list[Transition], list[int]]:
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def update_p_weights(self, indices_and_p_weights: list[tuple[int, float]]):
        pass


class ReplayBuffer(Buffer):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.memory = deque([], maxlen=capacity)

    def push(self, *args, p_weight: float):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        indices = sorted(random.sample(range(len(self.memory)), batch_size))
        return (
            [self.memory[i] for i in indices],
            indices,
            [1.0 for _ in range(batch_size)],
        )

    def __len__(self):
        return len(self.memory)

    def update_p_weights(self, indices_and_p_weights: list[tuple[int, float]]):
        return


@dataclass
class Item:
    p_weight: float
    transition: Transition

    def __le__(self, other):
        return self.p_weight <= other.p_weight

    def __lt__(self, other):
        return self.p_weight < other.p_weight

    def __ge__(self, other):
        return self.p_weight >= other.p_weight

    def __gt__(self, other):
        return self.p_weight > other.p_weight


@dataclass
class BucketInfo:
    start: int
    length: int
    weight: float


class PrioritizedReplayBuffer(Buffer):
    def __init__(
        self, capacity, num_buckets, *, sort_every_n: int = 1_000_000, beta: float = 0.9
    ):
        super().__init__(capacity)
        self.heap = []
        self.num_buckets = num_buckets
        self.bucket_infos = None
        self.total_weight = 0.0
        self.sort_every_n = sort_every_n
        self.num_update_since_last_sort = 0
        self.beta = beta

    def _initialize_bucket_starts_and_weights(self):
        self.bucket_infos = []
        self.total_weight = sum([1.0 / (i + 1) for i in range(len(self.heap))])
        per_bucket_weight = self.total_weight / float(self.num_buckets)
        current_start = 0
        current_length = 1
        current_weight = 1.0 / len(self.heap)
        for i in range(1, len(self.heap)):
            weight_delta = 1.0 / (len(self.heap) - i)
            if (
                len(self.heap) - i == self.num_buckets - len(self.bucket_infos) - 1
            ) or current_weight + weight_delta >= per_bucket_weight:
                self.bucket_infos.append(
                    BucketInfo(current_start, current_length, current_weight)
                )
                current_start, current_length, current_weight = i, 0, 0.0
            current_weight += weight_delta
            current_length += 1

        if current_length > 0:
            self.bucket_infos.append(
                BucketInfo(current_start, current_length, current_weight)
            )

        while len(self.bucket_infos) > self.num_buckets:
            bucket_info_1, bucket_info_2 = self.bucket_infos[-2], self.bucket_infos[-1]
            self.bucket_infos.pop()
            new_bucket_info = BucketInfo(
                bucket_info_1.start,
                bucket_info_1.length + bucket_info_2.length,
                bucket_info_1.weight + bucket_info_2.weight,
            )
            self.bucket_infos[-1] = new_bucket_info

    def _update_bucket_starts_and_weights(self):
        self.total_weight += 1.0 / len(self.heap)
        per_bucket_weight = self.total_weight / float(self.num_buckets)
        next_start_delta = 0
        next_weight_delta = 1.0 / len(self.heap)
        for i in range(self.num_buckets - 1):
            bucket_info = self.bucket_infos[i]
            start, length, weight = (
                bucket_info.start,
                bucket_info.length,
                bucket_info.weight,
            )

            if i == 0:
                length += 1
            else:
                start = start + 1 - next_start_delta
                length += next_start_delta
            weight += next_weight_delta

            next_start_delta = 0
            next_weight_delta = 0.0
            while (
                weight - 1.0 / (len(self.heap) - (start + length - 1))
                > per_bucket_weight
            ):
                weight_delta = 1.0 / (len(self.heap) - (start + length - 1))
                weight -= weight_delta
                next_weight_delta += weight_delta
                length -= 1
                next_start_delta += 1

            self.bucket_infos[i] = BucketInfo(start, length, weight)
        last_bucket_info = self.bucket_infos[-1]
        if len(self.bucket_infos) == 1:
            last_bucket_info.length += 1
        else:
            last_bucket_info.start += 1 - next_start_delta
            last_bucket_info.length += next_start_delta
        last_bucket_info.weight += next_weight_delta
        assert last_bucket_info.start + last_bucket_info.length == len(self.heap)

    def push(self, *args, p_weight: float):
        heappush(self.heap, Item(p_weight, Transition(*args)))
        if len(self.heap) > self.capacity:
            # TODO: Maybe popping the most stale one might be better?
            heappop(self.heap)
        elif self.bucket_infos is not None:
            self._update_bucket_starts_and_weights()
        self.num_update_since_last_sort += 1
        if self.num_update_since_last_sort % self.sort_every_n == 0:
            self.heap = sorted(self.heap)
            self.num_update_since_last_sort = 0

    def sample(self, batch_size):
        if self.bucket_infos is None:
            self._initialize_bucket_starts_and_weights()

        if self.num_buckets > batch_size:
            raise ValueError(f"{batch_size=} is smaller than {self.num_buckets=}.")

        bucket_indices = random.sample(range(self.num_buckets), batch_size)
        samples = []
        indices = []
        is_weights = []
        for bucket_idx in bucket_indices:
            bucket_info = self.bucket_infos[bucket_idx]
            heap_idx = random.randint(
                bucket_info.start, bucket_info.start + bucket_info.length - 1
            )
            samples.append(self.heap[heap_idx].transition)
            indices.append(heap_idx)
            # (N * P(j)) ** -beta
            is_weights.append(
                (
                    len(self.heap)
                    * (
                        (float(batch_size) / float(self.num_buckets))
                        * (1.0 / bucket_info.length)
                    )
                )
                ** (-1.0 * self.beta)
            )
        # Normalize importance sampling weights.
        max_is_weight = max(is_weights)
        is_weights = [w / max_is_weight for w in is_weights]
        return samples, indices, is_weights

    def __len__(self):
        return len(self.heap)

    def _parent_idx(self, index: int):
        return (index - 1) // 2

    def _decrease_p_weights(self, index: int, new_p_weight: float):
        item = self.heap[index]
        assert item.p_weight >= new_p_weight
        item.p_weight = new_p_weight
        while (
            index != 0 and self.heap[self._parent_idx(index)].p_weight > item.p_weight
        ):
            parent_idx = self._parent_idx(index)
            parent_item = self.heap[parent_idx]
            self.heap[parent_idx], self.heap[index] = item, parent_item
            index = parent_idx

    def update_p_weights(self, indices_and_p_weights: list[tuple[int, float]]):
        new_items = []
        # Sort by key to ensure that later indicies are preserved even after
        # removing previous items.
        indices_and_p_weights = sorted(indices_and_p_weights)
        for idx, new_p_weight in indices_and_p_weights:
            item = self.heap[idx]
            transition = item.transition
            new_items.append(Item(new_p_weight, transition))
            self._decrease_p_weights(idx, MIN_P_WEIGHT)

        # Pop all the items out.
        for _ in range(len(indices_and_p_weights)):
            assert heappop(self.heap).p_weight == MIN_P_WEIGHT

        for item in new_items:
            heappush(self.heap, item)

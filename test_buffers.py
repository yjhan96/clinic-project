import random

from buffers import PrioritizedReplayBuffer, ReplayBuffer, Transition


def get_dummy_transition(i: int) -> tuple:
    return (i, i, i, i, i)


def test_replay_buffer():
    random.seed(42)
    buffer = ReplayBuffer(10)
    for i in range(5):
        buffer.push(*get_dummy_transition(i), p_weight=2.0)

    assert buffer.sample(2) == (
        [Transition(0, 0, 0, 0, 0), Transition(4, 4, 4, 4, 4)],
        [0, 4],
        [1.0, 1.0],
    )

    for i in range(6):
        buffer.push(*get_dummy_transition(i), p_weight=2.0)

    assert len(buffer) == 10


def test_prioritized_replay_buffer():
    random.seed(42)
    batch_size = 5
    buffer = PrioritizedReplayBuffer(200, batch_size)
    # Add transitions with increasing p_weight.
    for i in range(100):
        buffer.push(*get_dummy_transition(i), p_weight=i + 1)
    assert len(buffer) == 100
    samples, indices, is_weights = buffer.sample(batch_size)
    assert indices == [28, 98, 88, 85, 97]
    assert max(is_weights) == 1.0
    indices_and_p_weights = [(idx, 0.0) for idx in indices]
    buffer.update_p_weights(indices_and_p_weights)
    assert sorted(
        [item.transition for item in sorted(buffer.heap)[:batch_size]]
    ) == sorted(samples)

    # Add more elements
    for i in range(100, 200):
        buffer.push(*get_dummy_transition(i), p_weight=i + 1)
    assert len(buffer) == 200
    samples, indices, _ = buffer.sample(batch_size)
    assert indices == [198, 23, 185, 153, 197]
    indices_and_p_weights = [(idx, -0.5) for idx in indices]
    buffer.update_p_weights(indices_and_p_weights)
    assert sorted(
        [item.transition for item in sorted(buffer.heap)[:batch_size]]
    ) == sorted(samples)

    buffer.push(*get_dummy_transition(200), p_weight=201)
    assert len(buffer) == 200


def test_prioritized_replay_buffer_sorted():
    random.seed(42)
    batch_size = 5
    buffer = PrioritizedReplayBuffer(100, batch_size, sort_every_n=100)
    # Add in reverse to make sure the heap was originally not sorted.
    for i in range(99, -1, -1):
        buffer.push(*get_dummy_transition(i), p_weight=i + 100)
    transitions = [item.transition for item in buffer.heap]
    assert transitions == [Transition(*get_dummy_transition(i)) for i in range(100)]


def test_prioritized_replay_buffer_num_buckets():
    random.seed(42)
    batch_size = 100
    buffer = PrioritizedReplayBuffer(batch_size + 1, batch_size)
    for i in range(101):
        buffer.push(*get_dummy_transition(i), p_weight=i + 1)

    buffer.sample(100)

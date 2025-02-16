from clinic_agent import ClinicAgent
from clinic_environment import ClinicEnv
import numpy as np


# TODO: Deduplicate.
def get_test_env() -> ClinicEnv:
    clinic_capacity = np.array([1, 2])
    clinic_travel_times = np.array([[0, 10], [10, 0]])
    patient_times = np.array([30, 40])
    num_nurses = 2
    clinic_env = ClinicEnv(
        clinic_capacity=clinic_capacity,
        clinic_travel_times=clinic_travel_times,
        patient_times=patient_times,
        num_nurses=num_nurses,
    )
    return clinic_env


def get_all_actions(action_space):
    def generator(actions, accum):
        if len(actions) == 0:
            yield accum
        else:
            action, rest = actions[0], actions[1:]
            for i in range(action.n):
                yield from generator(rest, accum + (i,))

    return generator(action_space.spaces, ())


def test_action_conversion():
    clinic_env = get_test_env()
    agent = ClinicAgent(
        clinic_env,
        learning_rate=1.0,
        initial_epsilon=0.2,
        epsilon_decay=0.05,
        final_epsilon=0.05,
    )
    for action in get_all_actions(clinic_env.action_space):
        assert agent._to_action(agent._from_action(action)) == action

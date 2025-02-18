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

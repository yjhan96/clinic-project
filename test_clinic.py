import numpy as np

from clinic_environment import ClinicEnv


def test_obs():
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

    obs, info = clinic_env.reset()
    assert info == None
    assert sorted(list(obs.keys())) == ["clinics", "nurses", "patients"]

    assert obs["clinics"] == (
        {"capacity": 1.0, "num_patients": 0.0},
        {"capacity": 2.0, "num_patients": 0.0},
    )

    assert obs["nurses"] == (
        {"location": 0, "operating_minutes_left": 0.0, "traveling_minutes_left": 0.0},
        {"location": 0, "operating_minutes_left": 0.0, "traveling_minutes_left": 0.0},
    )

    assert obs["patients"] == (
        {
            "status": 1,
            "treatment_time": 30.0,
            "minutes_in_treatment": 0.0,
            "treated_at": 0,
        },
        {
            "status": 1,
            "treatment_time": 40.0,
            "minutes_in_treatment": 0.0,
            "treated_at": 0,
        },
    )

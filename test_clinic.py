import numpy as np

from clinic_environment import ClinicEnv


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


def test_obs():
    clinic_env = get_test_env()
    (state, obs), info = clinic_env.reset()
    assert info == None
    assert state == 0
    assert sorted(list(obs.keys())) == ["clinics", "nurses", "patients"]

    assert obs["clinics"] == (
        {"capacity": 1.0, "num_patients": 0.0},
        {"capacity": 2.0, "num_patients": 0.0},
    )

    assert obs["nurses"] == (
        {"location": 1, "operating_minutes_left": 0.0, "traveling_minutes_left": 0.0},
        {"location": 1, "operating_minutes_left": 0.0, "traveling_minutes_left": 0.0},
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


def test_basic():
    clinic_env = get_test_env()

    (state, obs), reward, terminated, truncated, _ = clinic_env.step((1, 4))
    assert state == 0
    nurse_states = obs["nurses"]
    assert nurse_states[0] == {
        "location": 1,
        "operating_minutes_left": 10.0,
        "traveling_minutes_left": 0.0,
    }
    assert nurse_states[1] == {
        "location": 2,
        "operating_minutes_left": 0.0,
        "traveling_minutes_left": 5.0,
    }

    clinic_states = obs["clinics"]
    assert clinic_states[0] == {"capacity": 1.0, "num_patients": 1.0}

    patient_states = obs["patients"]
    assert patient_states[0] == {
        "status": 2,
        "treatment_time": 30.0,
        "minutes_in_treatment": 5.0,
        "treated_at": 1,
    }
    assert reward == 0
    assert not terminated and not truncated

    (_, obs), reward, _, _, _ = clinic_env.step((0, 0))
    nurse_states = obs["nurses"]
    assert nurse_states[0] == {
        "location": 1,
        "operating_minutes_left": 5.0,
        "traveling_minutes_left": 0.0,
    }
    assert nurse_states[1] == {
        "location": 2,
        "operating_minutes_left": 0.0,
        "traveling_minutes_left": 0.0,
    }
    patient_states = obs["patients"]
    assert patient_states[0] == {
        "status": 2,
        "treatment_time": 30.0,
        "minutes_in_treatment": 10.0,
        "treated_at": 1,
    }
    assert reward == 0

    (_, obs), reward, _, _, _ = clinic_env.step((0, 2))
    nurse_states = obs["nurses"]
    assert nurse_states[0] == {
        "location": 1,
        "operating_minutes_left": 0.0,
        "traveling_minutes_left": 0.0,
    }
    assert nurse_states[1] == {
        "location": 2,
        "operating_minutes_left": 10.0,
        "traveling_minutes_left": 0.0,
    }

    clinic_states = obs["clinics"]
    assert clinic_states[0] == {"capacity": 1.0, "num_patients": 1.0}
    assert clinic_states[1] == {"capacity": 2.0, "num_patients": 1.0}

    patient_states = obs["patients"]
    assert patient_states[0] == {
        "status": 2,
        "treatment_time": 30.0,
        "minutes_in_treatment": 15.0,
        "treated_at": 1,
    }
    assert patient_states[1] == {
        "status": 2,
        "treatment_time": 40.0,
        "minutes_in_treatment": 5.0,
        "treated_at": 2,
    }
    assert reward == 0

    (_, obs), reward, _, _, _ = clinic_env.step((1, 0))
    nurse_states = obs["nurses"]
    assert nurse_states[0] == {
        "location": 1,
        "operating_minutes_left": 10.0,
        "traveling_minutes_left": 0.0,
    }
    patient_states = obs["patients"]
    assert patient_states[0] == {
        "status": 2,
        "treatment_time": 30.0,
        "minutes_in_treatment": 20.0,
        "treated_at": 1,
    }

    clinic_env.step((0, 0))
    (_, obs), reward, _, _, _ = clinic_env.step((0, 0))

    nurse_states = obs["nurses"]
    assert nurse_states[0] == {
        "location": 1,
        "operating_minutes_left": 0.0,
        "traveling_minutes_left": 0.0,
    }

    clinic_states = obs["clinics"]
    assert clinic_states[0] == {"capacity": 1.0, "num_patients": 0.0}

    patient_states = obs["patients"]
    assert patient_states[0] == {
        "status": 3,
        "treatment_time": 30.0,
        "minutes_in_treatment": 30.0,
        "treated_at": 0,
    }
    assert patient_states[1] == {
        "status": 2,
        "treatment_time": 40.0,
        "minutes_in_treatment": 20.0,
        "treated_at": 2,
    }
    assert reward == 1

    clinic_env.step((0, 0))
    (_, obs), reward, _, _, _ = clinic_env.step((0, 2))

    patient_states = obs["patients"]
    assert patient_states[1] == {
        "status": 2,
        "treatment_time": 40.0,
        "minutes_in_treatment": 30.0,
        "treated_at": 2,
    }

    clinic_env.step((0, 0))
    (_, obs), reward, terminated, _, _ = clinic_env.step((0, 0))
    assert reward == 1
    assert terminated

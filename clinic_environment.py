from enum import IntEnum
import gymnasium as gym
import numpy as np

OPERATING_TIME = 15
MAX_TREATMENT_TIME = 1000
MAX_CAPACITY = 10


class NurseStatus(IntEnum):
    IDLE = 1
    IN_OPERATION = 2
    TRAVELING = 3


class Nurse:
    def __init__(self, location: int):
        self.location = location
        self.status: NurseStatus = NurseStatus.IDLE
        self.operating_minutes_left = 0
        self.traveling_minutes_left = 0

    @staticmethod
    def get_observation_space(num_clinics: int) -> gym.spaces.Space:
        return gym.spaces.Dict(
            {
                "location": gym.spaces.Discrete(num_clinics),
                "operating_minutes_left": gym.spaces.Box(0.0, float(OPERATING_TIME)),
                "traveling_minutes_left": gym.spaces.Box(0.0, float(OPERATING_TIME)),
            }
        )

    def _get_obs(self):
        return {
            "location": self.location,
            "operating_minutes_left": float(self.operating_minutes_left),
            "traveling_minutes_left": float(self.traveling_minutes_left),
        }


class PatientStatus(IntEnum):
    NOT_TREATED = 1
    IN_TREATMENT = 2
    DONE = 3


class Patient:
    def __init__(self, treatment_time: int):
        self.treatmet_time = treatment_time
        self.status: PatientStatus = PatientStatus.NOT_TREATED
        self.minutes_in_treatment: int = 0
        self.treated_at: "Clinic" | None = None

    @staticmethod
    def get_observation_space(num_clinics: int) -> gym.spaces.Space:
        return gym.spaces.Dict(
            {
                "status": gym.spaces.Discrete(3, start=1),
                "treatment_time": gym.spaces.Box(0.0, float(MAX_TREATMENT_TIME)),
                "minutes_in_treatment": gym.spaces.Box(0.0, float(MAX_TREATMENT_TIME)),
                "treated_at": gym.spaces.Discrete(num_clinics + 1),
            }
        )

    def _get_obs(self):
        return {
            "status": int(self.status),
            "treatment_time": float(self.treatmet_time),
            "minutes_in_treatment": float(self.minutes_in_treatment),
            "treated_at": 0 if self.treated_at is None else self.treated_at.idx,
        }


class Clinic:
    def __init__(self, idx, capacity):
        self.idx = idx
        self.capacity = capacity
        self.patients = []

    @staticmethod
    def get_observation_space() -> gym.spaces.Space:
        return gym.spaces.Dict(
            {
                "capacity": gym.spaces.Box(1.0, float(MAX_CAPACITY)),
                "num_patients": gym.spaces.Box(1.0, float(MAX_CAPACITY)),
            }
        )

    def _get_obs(self):
        return {
            "capacity": float(self.capacity),
            "num_patients": float(len(self.patients)),
        }


class ClinicEnv(gym.Env):
    def __init__(
        self,
        clinic_capacity: np.array,
        clinic_travel_times: np.ndarray,
        patient_times: np.array,
        num_nurses: int,
    ):
        self.clinic_capacity = clinic_capacity
        self.clinic_travel_times = clinic_travel_times
        self.patient_times = patient_times
        self.num_nurses = num_nurses

        self._reset()

        self.observation_space = gym.spaces.Dict(
            {
                "nurses": gym.spaces.Tuple(
                    tuple(
                        [
                            Nurse.get_observation_space(len(self.clinics))
                            for _ in self.nurses
                        ]
                    )
                ),
                "patients": gym.spaces.Tuple(
                    tuple(
                        [
                            Patient.get_observation_space(len(self.clinics))
                            for _ in self.patients
                        ]
                    )
                ),
                "clinics": gym.spaces.Tuple(
                    tuple([Clinic.get_observation_space() for _ in self.clinics])
                ),
            }
        )

        self.action_space = gym.spaces.Tuple(
            tuple(
                [
                    gym.spaces.MultiDiscrete(
                        [len(self.patients) + 1, len(self.clinics) + 1]
                    )
                ]
            )
        )

    def _get_obs(self):
        return {
            "nurses": tuple([nurse._get_obs() for nurse in self.nurses]),
            "patients": tuple([patient._get_obs() for patient in self.patients]),
            "clinics": tuple([clinic._get_obs() for clinic in self.clinics]),
        }

    def _get_info(self):
        return None

    def _reset(self):
        self.nurses = [Nurse(0) for _ in range(self.num_nurses)]
        self.patients = [Patient(patient_time) for patient_time in self.patient_times]
        self.clinics = [
            Clinic(idx, capacity)
            for idx, capacity in np.ndenumerate(self.clinic_capacity)
        ]

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset()
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

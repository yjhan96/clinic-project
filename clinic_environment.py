from enum import IntEnum

import gymnasium as gym
import numpy as np

OPERATING_TIME = 15
MAX_TRAVEL_TIME = 30
MAX_TREATMENT_TIME = 1000
MAX_CAPACITY = 10
MINUTE_PER_STEP = 5


class NurseStatus(IntEnum):
    IDLE = 1
    IN_OPERATION = 2
    TRAVELING = 3


class Nurse:
    def __init__(self, initial_clinic: "Clinic"):
        self.location: "Clinic" = initial_clinic
        self.status: NurseStatus = NurseStatus.IDLE
        self.operating_minutes_left: int = 0
        self.traveling_minutes_left: int = 0
        self.treating_patient: "Patient" | None = None

    @staticmethod
    def get_observation_space(num_clinics: int) -> gym.spaces.Space:
        return gym.spaces.Dict(
            {
                "location": gym.spaces.Discrete(num_clinics, start=1),
                "operating_minutes_left": gym.spaces.Box(0.0, float(OPERATING_TIME)),
                "traveling_minutes_left": gym.spaces.Box(0.0, float(OPERATING_TIME)),
                "status": gym.spaces.Discrete(3, start=1),
            }
        )

    @staticmethod
    def get_normalized_observation_space(num_clinics: int) -> gym.spaces.Space:
        return gym.spaces.Dict(
            {
                "location": gym.spaces.Discrete(num_clinics, start=1),
                "operating_minutes_left": gym.spaces.Box(0.0, 1.0),
                "traveling_minutes_left": gym.spaces.Box(0.0, 1.0),
                "status": gym.spaces.Discrete(3, start=1),
            }
        )

    def _get_obs(self):
        return {
            "location": self.location.idx + 1,
            "operating_minutes_left": float(self.operating_minutes_left),
            "traveling_minutes_left": float(self.traveling_minutes_left),
            "status": self.status,
        }

    @staticmethod
    def normalize_obs(obs: dict) -> dict:
        return {
            "location": obs["location"],
            "operating_minutes_left": obs["operating_minutes_left"]
            / float(OPERATING_TIME),
            "traveling_minutes_left": obs["traveling_minutes_left"]
            / float(MAX_TRAVEL_TIME),
            "status": obs["status"],
        }

    def treat(self, patient: "Patient"):
        assert self.status == NurseStatus.IDLE
        assert self.operating_minutes_left == 0
        assert self.treating_patient is None

        self.status = NurseStatus.IN_OPERATION
        self.operating_minutes_left = OPERATING_TIME
        self.treating_patient = patient

    def travel_to(self, clinic: "Clinic", traveling_time: int):
        assert self.status == NurseStatus.IDLE
        if clinic == self.location:
            assert traveling_time == 0
            return

        self.status = NurseStatus.TRAVELING
        self.traveling_minutes_left = traveling_time
        self.location = clinic

    def step(self, num_minutes):
        match self.status:
            case NurseStatus.IDLE:
                return
            case NurseStatus.IN_OPERATION:
                self.operating_minutes_left -= num_minutes
                if self.operating_minutes_left == 0:
                    self.status = NurseStatus.IDLE
                    self.treating_patient = None
            case NurseStatus.TRAVELING:
                self.traveling_minutes_left -= num_minutes
                if self.traveling_minutes_left == 0:
                    self.status = NurseStatus.IDLE
            case _:
                raise ValueError(f"Unknown nurse status: {self.status}")


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
        self.treating_nurse: Nurse | None = None

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

    @staticmethod
    def get_normalized_observation_space(num_clinics: int) -> gym.spaces.Space:
        return gym.spaces.Dict(
            {
                "status": gym.spaces.Discrete(3, start=1),
                "treatment_rate": gym.spaces.Box(0.0, 1.0),
                "minutes_in_treatment": gym.spaces.Box(0.0, 1.0),
                "treated_at": gym.spaces.Discrete(num_clinics + 1),
            }
        )

    def _get_obs(self):
        return {
            "status": int(self.status),
            "treatment_time": float(self.treatmet_time),
            "minutes_in_treatment": float(self.minutes_in_treatment),
            "treated_at": 0 if self.treated_at is None else self.treated_at.idx + 1,
        }

    @staticmethod
    def normalize_obs(obs: dict) -> dict:
        return {
            "status": obs["status"],
            "treatment_rate": 1.0 / obs["treatment_time"],
            "minutes_in_treatment": obs["minutes_in_treatment"] / obs["treatment_time"],
            "treated_at": obs["treated_at"],
        }

    def get_treatment_from(self, nurse: Nurse):
        if self.status == PatientStatus.NOT_TREATED:
            self.status = PatientStatus.IN_TREATMENT
            self.treating_nurse = nurse
            self.treated_at = nurse.location
            nurse.location.add_patient(self)
        elif self.status == PatientStatus.IN_TREATMENT:
            if self.treatmet_time - self.minutes_in_treatment == OPERATING_TIME:
                if self.treated_at == nurse.location:
                    self.treating_nurse = nurse
                else:
                    raise ValueError(
                        f"Nurse location ({nurse.location}) is different from patient location ({self.treated_at})"
                    )
            else:
                raise ValueError(
                    f"patient treatment time is incorrect: {self.treatmet_time=}, {self.minutes_in_treatment=}"
                )
        else:
            raise ValueError(f"Invalid status: {self.status}")

    def needs_nurse(self) -> bool:
        not_treated_yet = self.status == PatientStatus.NOT_TREATED
        needs_last_treatment = (
            self.status == PatientStatus.IN_TREATMENT
            and self.treatmet_time - self.minutes_in_treatment == OPERATING_TIME
        )
        return not_treated_yet or needs_last_treatment

    def step(self, time_in_minutes: int) -> bool:
        match self.status:
            case PatientStatus.NOT_TREATED:
                return True
            case PatientStatus.IN_TREATMENT:
                if (
                    self.minutes_in_treatment == self.treatmet_time - OPERATING_TIME
                    and self.treating_nurse is None
                ):
                    return False

                self.minutes_in_treatment += time_in_minutes
                if self.minutes_in_treatment == OPERATING_TIME:
                    self.treating_nurse = None
                elif self.minutes_in_treatment == self.treatmet_time:
                    self.status = PatientStatus.DONE
                    self.treating_nurse = None
                    clinic = self.treated_at
                    self.treated_at = None
                    clinic.remove_patient(self)
                return True
            case PatientStatus.DONE:
                return True
            case _:
                raise ValueError(f"Invalid status: {self.status}")


class Clinic:
    def __init__(self, idx: int, capacity: int):
        self.idx: int = idx
        self.capacity: int = capacity
        self.patients: list[Patient] = []

    @staticmethod
    def get_observation_space() -> gym.spaces.Space:
        return gym.spaces.Dict(
            {
                "capacity": gym.spaces.Box(1.0, float(MAX_CAPACITY)),
                "num_patients": gym.spaces.Box(1.0, float(MAX_CAPACITY)),
            }
        )

    @staticmethod
    def get_normalized_observation_space() -> gym.spaces.Space:
        return gym.spaces.Dict(
            {
                "fill_rate": gym.spaces.Box(0.0, 1.0),
                "fill_percentage": gym.spaces.Box(0.0, 1.0),
            }
        )

    def _get_obs(self):
        return {
            "capacity": float(self.capacity),
            "num_patients": float(len(self.patients)),
        }

    @staticmethod
    def normalize_obs(obs: dict) -> dict:
        return {
            "fill_rate": 1.0 / obs["capacity"],
            "fill_percentage": obs["num_patients"] / obs["capacity"],
        }

    def num_available_seats(self) -> int:
        return self.capacity - len(self.patients)

    def add_patient(self, patient: Patient):
        if self.num_available_seats() == 0:
            raise ValueError("Clinic is full!")
        self.patients.append(patient)

    def remove_patient(self, done_patient: Patient):
        self.patients = [
            patient for patient in self.patients if patient != done_patient
        ]


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

        self._nonterminal_observation_space = gym.spaces.Dict(
            {
                "nurse_turn": gym.spaces.Discrete(num_nurses),
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
        self.nonterminal_normalized_observation_space = gym.spaces.Dict(
            {
                "nurse_turn": gym.spaces.Discrete(num_nurses),
                "nurses": gym.spaces.Tuple(
                    tuple(
                        [
                            Nurse.get_normalized_observation_space(len(self.clinics))
                            for _ in self.nurses
                        ]
                    )
                ),
                "patients": gym.spaces.Tuple(
                    tuple(
                        [
                            Patient.get_normalized_observation_space(len(self.clinics))
                            for _ in self.patients
                        ]
                    )
                ),
                "clinics": gym.spaces.Tuple(
                    tuple(
                        [
                            Clinic.get_normalized_observation_space()
                            for _ in self.clinics
                        ]
                    )
                ),
            }
        )
        self._terminal_observation_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.OneOf(
            (self._nonterminal_observation_space, self._terminal_observation_space)
        )
        self.action_space = gym.spaces.Discrete(
            1 + len(self.patients) + len(self.clinics)
        )

    def _get_obs(self):
        if not self.is_terminated:
            return (
                0,
                {
                    "nurse_turn": self.nurse_turn,
                    "nurses": tuple([nurse._get_obs() for nurse in self.nurses]),
                    "patients": tuple(
                        [patient._get_obs() for patient in self.patients]
                    ),
                    "clinics": tuple([clinic._get_obs() for clinic in self.clinics]),
                },
            )
        else:
            return (1, 0)

    def _get_info(self):
        return {}

    def _reset(self):
        self.clinics = [
            Clinic(idx[0], capacity)
            for idx, capacity in np.ndenumerate(self.clinic_capacity)
        ]
        self.nurses = [Nurse(self.clinics[0]) for _ in range(self.num_nurses)]
        self.patients = [Patient(patient_time) for patient_time in self.patient_times]
        self.is_terminated = False
        # Index of which nurse to take action.
        self.nurse_turn = 0
        self.num_turns = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset()
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def is_valid_treatment(self, nurse: Nurse, patient: Patient) -> bool:
        """Returns True iff nurse can treat patient at a valid time."""
        # Nurse needs to be idle, and patient must seek for a nurse.
        if not (nurse.status == NurseStatus.IDLE and patient.needs_nurse()):
            return False

        if patient.treated_at is not None and patient.treated_at != nurse.location:
            return False

        if patient.treated_at is None and nurse.location.num_available_seats() == 0:
            return False

        return True

    def is_valid_travel(self, nurse: Nurse, clinic: Clinic) -> bool:
        return nurse.status == NurseStatus.IDLE

    def _get_terminal_state(self):
        self.is_terminated = True
        obs = self._get_obs()
        reward = -1
        terminated = True
        truncated = False
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def get_valid_actions(self) -> tuple[int]:
        nurse = self.nurses[self.nurse_turn]
        valid_nurse_actions = []
        for nurse_action in range(self.action_space.n):
            if self._is_valid_nurse_action(nurse, nurse_action):
                valid_nurse_actions.append(nurse_action)

        return tuple(valid_nurse_actions)

    def _is_valid_nurse_action(self, nurse: Nurse, nurse_action: int) -> bool:
        if nurse_action == 0:
            return True
        elif nurse_action >= 1 and nurse_action <= len(self.patients):
            patient_idx = nurse_action - 1
            patient_to_treat = self.patients[patient_idx]
            if not self.is_valid_treatment(nurse, patient_to_treat):
                return False
        else:
            dest_clinic = nurse_action - len(self.patients) - 1
            clinic = self.clinics[dest_clinic]
            if not self.is_valid_travel(nurse, clinic):
                return False
        return True

    def step(self, nurse_action: tuple[int]):
        nurse = self.nurses[self.nurse_turn]
        if nurse_action == 0:
            pass
        elif nurse_action >= 1 and nurse_action <= len(self.patients):
            patient_idx = nurse_action - 1
            patient_to_treat = self.patients[patient_idx]
            if self.is_valid_treatment(nurse, patient_to_treat):
                patient_to_treat.get_treatment_from(nurse)
                nurse.treat(patient_to_treat)
            else:
                return self._get_terminal_state()
        else:
            dest_clinic = nurse_action - len(self.patients) - 1
            clinic = self.clinics[dest_clinic]
            if self.is_valid_travel(nurse, clinic):
                nurse.travel_to(
                    clinic, self.clinic_travel_times[nurse.location.idx][clinic.idx]
                )
            else:
                return self._get_terminal_state()

        if self.nurse_turn == self.num_nurses - 1:
            self.nurse_turn = 0
            self.num_turns += 1
            # Step through time.
            for nurse in self.nurses:
                nurse.step(MINUTE_PER_STEP)

            sick_patients = []
            for patient in self.patients:
                is_patient_ok = patient.step(MINUTE_PER_STEP)
                if not is_patient_ok:
                    sick_patients.append(patient)

            if len(sick_patients) > 0:
                return self._get_terminal_state()

            curr_num_done_patients = len(
                [
                    patient
                    for patient in self.patients
                    if patient.status == PatientStatus.DONE
                ]
            )

            obs = self._get_obs()
            info = self._get_info()
            # Add minus one to encourage an agent to finish early.
            terminated = curr_num_done_patients == len(self.patients)
            reward = 1.0 / self.num_turns if terminated else 0.0
            truncated = False

            return obs, reward, terminated, truncated, info
        else:
            self.nurse_turn += 1
            obs = self._get_obs()
            info = self._get_info()
            reward = 0
            terminated = False
            truncated = False

            return obs, reward, terminated, truncated, info

    def normalize_state(self, state: dict) -> np.array:
        mode, obs = state
        if mode == 1:
            return None

        normalized_obs = {}
        normalized_obs["nurse_turn"] = obs["nurse_turn"]
        normalized_obs["nurses"] = tuple(
            [Nurse.normalize_obs(nurse_obs) for nurse_obs in obs["nurses"]]
        )
        normalized_obs["patients"] = tuple(
            [Patient.normalize_obs(patient_obs) for patient_obs in obs["patients"]]
        )
        normalized_obs["clinics"] = tuple(
            [Clinic.normalize_obs(clinic_obs) for clinic_obs in obs["clinics"]]
        )
        return normalized_obs

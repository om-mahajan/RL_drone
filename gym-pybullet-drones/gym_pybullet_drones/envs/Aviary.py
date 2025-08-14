import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class Aviary(BaseRLAviary):
    def __init__(self,
                drone_model: DroneModel=DroneModel.CF2X,
                initial_xyzs=None,
                initial_rpys=None,
                physics: Physics=Physics.PYB,
                pyb_freq: int = 240,
                ctrl_freq: int = 30,
                gui=False,
                record=False,
                obs: ObservationType=ObservationType.KIN,
                act: ActionType=ActionType.RPM
                ):
        self.TARGET_POS = np.array([1,0,1])
        self.EPISODE_LEN_SEC = 8

        super().__init__(drone_model=drone_model,
                    num_drones=1,
                    initial_xyzs=initial_xyzs,
                    initial_rpys=initial_rpys,
                    physics=physics,
                    pyb_freq=pyb_freq,
                    ctrl_freq=ctrl_freq,
                    gui=gui,
                    record=record,
                    obs=obs,
                    act=act
                    )
    def _computeReward(self):
        state = self._getDroneStateVector(0)
        ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
        return ret
    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
            return True
        else:
            return False
    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
        
    def _computeInfo(self):
        return {"target_pos": self.TARGET_POS,
                "drone_pos": self._getDroneStateVector(0)[0:3],
                "step_counter": self.step_counter,
                "episode_len_sec": self.EPISODE_LEN_SEC,
                "episode_len_steps": self.EPISODE_LEN_SEC * self.PYB_FREQ}
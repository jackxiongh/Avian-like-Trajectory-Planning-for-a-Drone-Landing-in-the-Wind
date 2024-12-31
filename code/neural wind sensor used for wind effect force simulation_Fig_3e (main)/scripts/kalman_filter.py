import numpy as np


class KalmanAdaptive:
    def __init__(self):
        self.dt = 0.02
        self.lamda = 0.05
        self.R = 60.0 * np.eye(3)
        self.Q = 1.0 * np.eye(9)
        self.P = 2.0 * np.eye(9)
        self.a = np.zeros((9, 1))

    def update(self, y_measurement, s, phi_output):
        phi = np.zeros((3, 9))
        phi[0, :3] = phi_output.T
        phi[1, 3:6] = phi_output.T
        phi[2, 6:9] = phi_output.T

        a_minus = (1 - self.lamda * self.dt) * self.a
        P_minus = (1 - self.lamda * self.dt) ** 2 * self.P + self.Q * self.dt
        K_temp = P_minus @ phi.T @ np.linalg.inv(phi @ P_minus @ phi.T + self.R * self.dt)
        a_plus = a_minus - K_temp @ (phi @ a_minus - y_measurement) - P_minus @ phi.T @ s
        self.P = (np.eye(9) - K_temp @ phi) @ P_minus @ (
                np.eye(9) - K_temp @ phi).T + self.dt * K_temp @ self.R @ K_temp.T
        self.a = a_plus

    def get_a(self):
        return self.a


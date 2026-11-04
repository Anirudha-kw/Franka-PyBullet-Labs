"""Joint impedance control of the Franka FR3 in NVIDIA Newton (runs on GPU via Warp, or on CPU).

    python newton_sim/joint_impedance.py                       # one robot, default device
    python newton_sim/joint_impedance.py --num-worlds 1024     # 1024 robots in parallel on the GPU

Right-click and drag a link in the viewer to pull on the robot and test compliance.
"""

import numpy as np
import warp as wp

import newton.examples
from franka_scene import ARM_DOF, Q_NEUTRAL, FrankaTorqueSim, create_parser

# Joint-space impedance: one torsional spring-damper per joint (Nm/rad, Nm·s/rad)
# Stiffer at the shoulder, softer towards the wrist
STIFFNESS = np.array([100.0, 100.0, 100.0, 100.0, 40.0, 30.0, 10.0])
DAMPING = 2.0 * np.sqrt(STIFFNESS)


@wp.kernel
def joint_impedance_kernel(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    q_d: wp.array(dtype=float),
    K: wp.array(dtype=float),
    D: wp.array(dtype=float),
    gravity_force: wp.array(dtype=float),
    coriolis_force: wp.array(dtype=float),
    dofs_per_world: int,
    tau_d: wp.array2d(dtype=float),
):
    w, j = wp.tid()  # one thread per (robot, joint)
    i = w * dofs_per_world + j
    # tau = K*(q_d - q) - D*dq + gravity + Coriolis
    tau_d[w, j] = K[j] * (q_d[j] - joint_q[i]) - D[j] * joint_qd[i] + gravity_force[i] + coriolis_force[i]


class JointImpedance(FrankaTorqueSim):
    def setup_controller(self):
        self.K = wp.array(STIFFNESS, dtype=float)
        self.D = wp.array(DAMPING, dtype=float)
        # Desired State (Equilibrium is the starting joint configuration)
        self.q_d = wp.array(Q_NEUTRAL, dtype=float)

    def compute_torques(self):
        wp.launch(
            joint_impedance_kernel,
            dim=(self.num_worlds, ARM_DOF),
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.q_d,
                self.K,
                self.D,
                self.gravity_force,
                self.coriolis_force,
                self.dofs_per_world,
            ],
            outputs=[self.tau_d],
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init(create_parser())
    newton.examples.run(JointImpedance(viewer, args), args)

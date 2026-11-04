"""Cartesian impedance control of the Franka FR3 in NVIDIA Newton (runs on GPU via Warp, or on CPU).

    python newton_sim/cartesian_impedance.py                       # one robot, default device
    python newton_sim/cartesian_impedance.py --num-worlds 1024     # 1024 robots in parallel on the GPU

Right-click and drag a link in the viewer to pull on the robot and test compliance.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from franka_scene import ARM_DOF, FrankaTorqueSim, create_parser

# Task-space impedance: [x, y, z] in N/m, [rx, ry, rz] in Nm/rad; critically damped for unit mass
STIFFNESS = np.array([150.0, 150.0, 150.0, 10.0, 10.0, 10.0])
DAMPING = 2.0 * np.sqrt(STIFFNESS)


@wp.func
def orientation_error(R: wp.mat33, R_d: wp.mat33) -> wp.vec3:
    """Rotation error in the base frame, as the vector part of the relative quaternion R^T R_d."""
    R_rel = wp.transpose(R) @ R_d
    trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
    s = 0.5 / wp.sqrt(wp.max(trace + 1.0, 1.0e-6))
    err_local = s * wp.vec3(R_rel[2, 1] - R_rel[1, 2], R_rel[0, 2] - R_rel[2, 0], R_rel[1, 0] - R_rel[0, 1])
    return -(R @ err_local)  # map error from the local EE frame back to the base frame


@wp.kernel
def init_desired_pose(
    body_q: wp.array(dtype=wp.transform),
    ee_body: int,
    bodies_per_world: int,
    pos_d: wp.array(dtype=wp.vec3),
    rot_d: wp.array(dtype=wp.mat33),
):
    w = wp.tid()
    X = body_q[w * bodies_per_world + ee_body]
    pos_d[w] = wp.transform_get_translation(X)
    rot_d[w] = wp.quat_to_matrix(wp.transform_get_rotation(X))


@wp.kernel
def cartesian_impedance_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    J: wp.array3d(dtype=float),
    joint_qd: wp.array(dtype=float),
    pos_d: wp.array(dtype=wp.vec3),
    rot_d: wp.array(dtype=wp.mat33),
    K: wp.array(dtype=float),
    D: wp.array(dtype=float),
    gravity_force: wp.array(dtype=float),
    coriolis_force: wp.array(dtype=float),
    ee_body: int,
    bodies_per_world: int,
    dofs_per_world: int,
    tau_d: wp.array2d(dtype=float),
):
    w = wp.tid()  # one thread per robot
    b = w * bodies_per_world + ee_body
    X = body_q[b]
    pos = wp.transform_get_translation(X)
    R = wp.quat_to_matrix(wp.transform_get_rotation(X))

    # Newton's Jacobian maps dq to the link's centre-of-mass velocity; shift it to the EE point:
    # v_ee = v_com + w x r, with r pointing from the COM to the EE frame origin
    r = -(R @ body_com[b])
    row = ee_body * 6

    # 1. Cartesian error and velocity (dx = J * dq)
    e_pos = pos - pos_d[w]
    e_rot = orientation_error(R, rot_d[w])
    v = wp.vec3(0.0)
    omega = wp.vec3(0.0)
    for j in range(ARM_DOF):
        Jv = wp.vec3(J[w, row + 0, j], J[w, row + 1, j], J[w, row + 2, j])
        Jw = wp.vec3(J[w, row + 3, j], J[w, row + 4, j], J[w, row + 5, j])
        qd = joint_qd[w * dofs_per_world + j]
        v += (Jv + wp.cross(Jw, r)) * qd
        omega += Jw * qd

    # 2. Control Law: F_task = -K*e - D*dx (diagonal gains)
    F = wp.vec3(
        -K[0] * e_pos[0] - D[0] * v[0], -K[1] * e_pos[1] - D[1] * v[1], -K[2] * e_pos[2] - D[2] * v[2]
    )
    M = wp.vec3(
        -K[3] * e_rot[0] - D[3] * omega[0], -K[4] * e_rot[1] - D[4] * omega[1], -K[5] * e_rot[2] - D[5] * omega[2]
    )

    # 3. Torque mapping tau_task = J^T F_task, plus gravity and Coriolis compensation
    for j in range(ARM_DOF):
        Jv = wp.vec3(J[w, row + 0, j], J[w, row + 1, j], J[w, row + 2, j])
        Jw = wp.vec3(J[w, row + 3, j], J[w, row + 4, j], J[w, row + 5, j])
        i = w * dofs_per_world + j
        tau_d[w, j] = wp.dot(Jv + wp.cross(Jw, r), F) + wp.dot(Jw, M) + gravity_force[i] + coriolis_force[i]


class CartesianImpedance(FrankaTorqueSim):
    def setup_controller(self):
        self.K = wp.array(STIFFNESS, dtype=float)
        self.D = wp.array(DAMPING, dtype=float)
        self.J = newton.eval_jacobian(self.model, self.state_0)
        self.joint_S_s = wp.zeros(self.model.joint_dof_count, dtype=wp.spatial_vector)

        # Desired State (Equilibrium is the starting EE pose of every robot)
        self.pos_d = wp.zeros(self.num_worlds, dtype=wp.vec3)
        self.rot_d = wp.zeros(self.num_worlds, dtype=wp.mat33)
        wp.launch(
            init_desired_pose,
            dim=self.num_worlds,
            inputs=[self.state_0.body_q, self.ee_body, self.bodies_per_world],
            outputs=[self.pos_d, self.rot_d],
        )

    def compute_torques(self):
        # Kinematics: Jacobians of every link of every robot
        newton.eval_jacobian(self.model, self.state_0, J=self.J, joint_S_s=self.joint_S_s)
        wp.launch(
            cartesian_impedance_kernel,
            dim=self.num_worlds,
            inputs=[
                self.state_0.body_q,
                self.model.body_com,
                self.J,
                self.state_0.joint_qd,
                self.pos_d,
                self.rot_d,
                self.K,
                self.D,
                self.gravity_force,
                self.coriolis_force,
                self.ee_body,
                self.bodies_per_world,
                self.dofs_per_world,
            ],
            outputs=[self.tau_d],
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init(create_parser())
    newton.examples.run(CartesianImpedance(viewer, args), args)

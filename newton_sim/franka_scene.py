"""Shared Newton scene for the Franka controllers.

Builds one torque-controlled Franka FR3 arm per simulated world, steps the physics at
1 kHz on any Warp device (CUDA GPU or CPU), and provides the gravity and Coriolis
torques every controller needs. A controller subclasses FrankaTorqueSim and implements
`compute_torques()`, which writes the desired arm torques into `self.tau_d`.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode

ARM_DOF = 7
Q_NEUTRAL = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]  # same neutral pose as the PyBullet scripts
FINGER_POS = 0.02            # [m] fingers held half-open
EE_BODY = "fr3_hand_tcp"     # point between the fingers (PyBullet: panda_grasptarget)
MAX_TORQUE_RATE = 1000.0     # [Nm/s] Franka's hardware limit


def create_parser():
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=1, help="Number of robots simulated in parallel.")
    parser.add_argument(
        "--solver", choices=["mujoco", "featherstone"], default="mujoco", help="Newton physics solver."
    )
    return parser


@wp.kernel
def rate_limit_torques(
    tau_d: wp.array2d(dtype=float),
    tau_last: wp.array2d(dtype=float),
    max_delta: float,
    dofs_per_world: int,
    joint_f: wp.array(dtype=float),
):
    # Safety: limit how fast the commanded torque may change (1000 Nm/s -> 1 Nm per 1 ms step)
    w, j = wp.tid()
    prev = tau_last[w, j]
    tau = prev + wp.clamp(tau_d[w, j] - prev, -max_delta, max_delta)
    tau_last[w, j] = tau
    joint_f[w * dofs_per_world + j] = tau


class FrankaTorqueSim:
    def __init__(self, viewer, args):
        # 1kHz control loop: 20 physics/control steps of 1 ms per 50 Hz rendered frame
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 20
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()
        self.num_worlds = args.num_worlds

        # Load Robot (Franka FR3 from Newton's asset library)
        robot = newton.ModelBuilder()
        urdf = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
        robot.add_urdf(str(urdf), floating=False, enable_self_collisions=False, joint_damping_mode="passive")
        self.ee_body = next(i for i, label in enumerate(robot.body_label) if label.endswith(EE_BODY))
        self.bodies_per_world = robot.body_count
        self.dofs_per_world = robot.joint_dof_count

        # Initial Setup: Neutral Pose; arm joints driven purely by torque (joint_f)
        robot.joint_q[:ARM_DOF] = Q_NEUTRAL
        for i in range(ARM_DOF):
            robot.joint_target_mode[i] = int(JointTargetMode.EFFORT)
            robot.joint_target_ke[i] = 0.0
            robot.joint_target_kd[i] = 0.0
        # Gripper fingers: held in place by a position servo
        for i in range(ARM_DOF, self.dofs_per_world):
            robot.joint_q[i] = FINGER_POS
            robot.joint_target_q[i] = FINGER_POS
            robot.joint_target_mode[i] = int(JointTargetMode.POSITION)
            robot.joint_target_ke[i] = 500.0
            robot.joint_target_kd[i] = 20.0

        # One robot per world, laid out on a grid
        scene = newton.ModelBuilder()
        scene.replicate(robot, self.num_worlds, spacing=(1.2, 1.2, 0.0))
        scene.add_ground_plane()  # visual only: the solvers run without contacts
        self.model = scene.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.ee_start = self.ee_positions()

        if args.solver == "mujoco":
            # MuJoCo Warp: all worlds stepped as one batch on the GPU (also runs on Warp's CPU backend)
            self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)
        else:
            self.solver = newton.solvers.SolverFeatherstone(self.model)

        # Buffers shared by all controllers
        n = self.model.joint_dof_count
        self.gravity_force = wp.zeros(n, dtype=float)
        self.coriolis_force = wp.zeros(n, dtype=float)
        self.tau_d = wp.zeros((self.num_worlds, ARM_DOF), dtype=float)
        self.tau_last = wp.zeros((self.num_worlds, ARM_DOF), dtype=float)
        self.max_delta_tau = MAX_TORQUE_RATE * self.sim_dt
        self.setup_controller()

        # Camera: aimed at the middle of all robots, pulled back as the grid of worlds grows
        self.viewer.set_model(self.model)
        bases = self.state_0.body_q.numpy()[:: self.bodies_per_world, :3]
        target = bases.mean(axis=0) + np.array([0.2, 0.0, 0.4])
        grid = np.ptp(bases[:, :2], axis=0).max()
        eye = target + np.array([0.8, 0.8, 0.4]) * (1.0 + grid / 0.6) + np.array([0.0, 0.0, 0.8 * grid])
        self.viewer.set_camera(pos=wp.vec3(*eye))
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.look_at(wp.vec3(*target))
        self.capture()

    def setup_controller(self):
        """Allocate gains and desired states (implemented by each controller)."""

    def compute_torques(self):
        """Write desired arm torques, including gravity and Coriolis terms, into self.tau_d."""
        raise NotImplementedError

    def capture(self):
        # On a GPU, record one frame of simulation as a CUDA graph and replay it every frame
        self.graph = None
        if self.device.is_cuda:
            state_0, state_1 = self.state_0, self.state_1
            try:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph
            except Exception as e:
                self.state_0, self.state_1 = state_0, state_1
                print(f"CUDA graph capture unavailable, running without it: {e}")

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)  # mouse picking in the viewer

            # Dynamics: gravity g(q) and Coriolis C(q, dq)dq for every robot
            newton.eval_inverse_dynamics_passive(
                self.model, self.state_0, gravity_force=self.gravity_force, coriolis_force=self.coriolis_force
            )
            self.compute_torques()
            wp.launch(
                rate_limit_torques,
                dim=(self.num_worlds, ARM_DOF),
                inputs=[self.tau_d, self.tau_last, self.max_delta_tau, self.dofs_per_world],
                outputs=[self.control.joint_f],
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def ee_positions(self):
        return self.state_0.body_q.numpy()[self.ee_body :: self.bodies_per_world, :3]

    def test_final(self):
        # --test: with nobody pushing, every robot must still hold its hand where it started
        err = np.linalg.norm(self.ee_positions() - self.ee_start, axis=1).max()
        assert err < 0.01, f"end-effector drifted {1000 * err:.1f} mm from its start position"

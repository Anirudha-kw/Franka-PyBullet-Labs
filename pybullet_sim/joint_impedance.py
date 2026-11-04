import pybullet as p
import pybullet_data
import numpy as np
import time

class FrankaJointImpedanceController:
    def __init__(self):
        # Initialize PyBullet
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # 1kHz control loop
        self.dt = 1.0/1000.0
        p.setTimeStep(self.dt)

        # Load Robot
        self.panda = p.loadURDF("franka_panda/panda.urdf", [0,0,0], useFixedBase=True)
        self.ee_link = 11

        # Identify all movable joints (usually 9: 7 for arm, 2 for gripper)
        self.movable_joints = []
        for i in range(p.getNumJoints(self.panda)):
            info = p.getJointInfo(self.panda, i)
            if info[2] != p.JOINT_FIXED:
                self.movable_joints.append(i)

        self.num_dof = len(self.movable_joints) # Total DoF for the URDF (9)
        self.arm_dof = 7                        # DoF for the Arm only

        # Joint-space impedance: one torsional spring-damper per joint (Nm/rad, Nm·s/rad)
        # Stiffer at the shoulder, softer towards the wrist
        self.stiffness_diag = np.array([100.0, 100.0, 100.0, 100.0, 40.0, 30.0, 10.0])
        self.damping_diag = 2.0 * np.sqrt(self.stiffness_diag)

        self.K = np.diag(self.stiffness_diag)
        self.D = np.diag(self.damping_diag)

        # Initial Setup: Neutral Pose
        self.q_neutral = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        for i in range(self.arm_dof):
            p.resetJointState(self.panda, self.movable_joints[i], self.q_neutral[i])

        # Disable default velocity motors to enable torque control
        p.setJointMotorControlArray(self.panda, self.movable_joints, p.VELOCITY_CONTROL, forces=[0]*self.num_dof)

        # Desired State (Equilibrium is the starting joint configuration)
        self.q_d = self.q_neutral.copy()

        # Safety: Torque rate limiting (1000 Nm/s is Franka's hardware limit)
        self.tau_last = np.zeros(self.arm_dof)
        self.max_delta_tau = 1000.0 * self.dt # Max change allowed per 1ms step

    def get_ee_state(self):
        state = p.getLinkState(self.panda, self.ee_link)
        return np.array(state[0]), np.array(state[1])

    def get_dynamics(self, q_all, dq_all):
        """
        Computes Gravity and Coriolis terms separately.
        PyBullet requires full DoF lists for these calculations.
        """
        # Gravity compensation: Inverse dynamics with zero velocity/acceleration
        tau_grav = np.array(p.calculateInverseDynamics(self.panda, list(q_all), [0.0]*self.num_dof, [0.0]*self.num_dof))
        # Total dynamics: Grav + Coriolis/Centrifugal
        tau_total = np.array(p.calculateInverseDynamics(self.panda, list(q_all), list(dq_all), [0.0]*self.num_dof))

        # Return only the arm components (first 7)
        return tau_grav[:self.arm_dof], (tau_total - tau_grav)[:self.arm_dof]

    def step(self):
        # 1. Sense: Get full state for all joints (including gripper)
        joint_states = p.getJointStates(self.panda, self.movable_joints)
        q_all = np.array([s[0] for s in joint_states])
        dq_all = np.array([s[1] for s in joint_states])

        # Subsets for the 7-DOF arm
        q = q_all[:self.arm_dof]
        dq = dq_all[:self.arm_dof]

        # 2. Control Law (no Jacobian needed: errors are measured directly in joint space)
        # tau_task = K*(q_d - q) - D*dq
        tau_task = self.K @ (self.q_d - q) - self.D @ dq

        # 3. Dynamics: Add Gravity (for Sim) and Coriolis
        tau_grav, tau_coriolis = self.get_dynamics(q_all, dq_all)
        tau_d = tau_task + tau_coriolis + tau_grav

        # 4. Apply Torque Rate Limiting (Safety/Hardware Fidelity)
        diff = np.clip(tau_d - self.tau_last, -self.max_delta_tau, self.max_delta_tau)
        tau_saturated = self.tau_last + diff
        self.tau_last = tau_saturated

        # 5. Execute: Apply torques only to the arm
        p.setJointMotorControlArray(
            self.panda,
            self.movable_joints[:self.arm_dof],
            p.TORQUE_CONTROL,
            forces=tau_saturated
        )

        p.stepSimulation()

if __name__ == "__main__":
    controller = FrankaJointImpedanceController()

    print("Use Ctrl+Mouse to pull on the robot and test compliance.")

    try:
        while True:
            controller.step()
            time.sleep(controller.dt)
    except KeyboardInterrupt:
        p.disconnect()

import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import to_torch, quaternion_to_matrix, tensor_clamp,quat_diff_rad
from isaacgym.torch_utils import quat_conjugate,quat_mul,get_euler_xyz,quat_from_euler_xyz

class MagneticUr5(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.damping = 0.15
        self.max_episode_length = 600

        # self.cfg["env"]["numObservations"] = 14
        self.cfg["env"]["numObservations"] = 21
        self.cfg["env"]["numActions"] = 6

        self.debug_viz = True

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # 获取关节状态张量、根刚体状态张量、全部刚体的状态张量
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # 更新相关张量
        self.refresh_tensor()

        # 将关节状态张量,根刚体张量转换为torch.tensor
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)

        # 获取ur5对应的关节角度和角速度
        self.ur5_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_ur5_dofs]
        self.ur5_dof_pos = self.ur5_dof_state[...,0]
        self.ur5_dof_vel = self.ur5_dof_state[...,1]

        # 初始化ur5关节目标角度张量，ur5初始角度
        self.ur5_dof_targets = torch.zeros((self.num_envs,self.num_ur5_dofs),dtype=torch.float,device=self.device)
        self.ur5_default_dof_pos = to_torch([-1.5436e-02,  1.2114e-01,  1.0015e+00, -1.1235e+00, -1.5862e+00,
         -1.2507e-05], device=self.device)

        # 设置ur5的目标位置和姿态
        self.ur5_ee_pos_goal = torch.zeros_like(self.rigid_body_states[:,self.ur5_ee_handle][:, 0:3])
        self.ur5_ee_rot_goal = torch.zeros_like(self.rigid_body_states[:,self.ur5_ee_handle][:, 3:7])
        self.ur5_ee_pos_goal[:,] = to_torch([0.5,0.1,0.6],device=self.device)
        self.ur5_ee_rot_goal[:,] = to_torch([0.0000, 0.7071, 0.0000, 0.7071],device=self.device)

        # 单个环境里actor的总数量以及刚体的数量
        self.num_props = 5 
        self.fixed_obj = 4 

        # 全局索引
        self.global_indices = torch.arange(self.num_envs * self.num_props, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        # 由于以下物体是刚体没有关节，需单独处理
        self.capsule_states = self.root_state_tensor[:, self.num_props-self.fixed_obj:self.num_props-3]
        self.magnet_states = self.root_state_tensor[:, self.num_props-3:self.num_props-2]
        self.tank1_states = self.root_state_tensor[:, self.num_props-2:self.num_props-1]
        self.capsule_virtual_states = self.root_state_tensor[:,self.num_props-1:]

        # 设置磁体和胶囊的磁矩大小，设置平衡时的距离以及净重力
        self.moment_source_norm = 26.2
        self.moment_capsule_norm = 0.126
        self.magnet_balance_dis = torch.zeros((self.num_envs,3),device=self.device)
        self.magnet_balance_dis[:] = torch.tensor([0,0,-0.25],device=self.device)
        self.net_weight = torch.tensor([0.0,0.0,-0.0005],device=self.device)

        # 初始化胶囊的目标位置,胶囊的初始位置和平衡位置
        self.target_pos = torch.zeros((self.num_envs,3),device=self.device)
        self.target_rot = torch.zeros((self.num_envs,4),device=self.device)
        self.capsule_start_pos = torch.tensor([0.5,0.1,0.325],device=self.device)


        # reset/初始化
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs,self.cfg["env"]['envSpacing'],int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.restitution = 1 # 恢复系数
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_envs(self,num_envs,spacing,num_per_row):
        lower = gymapi.Vec3(-0.5*spacing, -0.5*spacing, 0.0)
        upper = gymapi.Vec3(0.5*spacing, 0.5*spacing, 0.5*spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        ur5_asset_file = "urdf/ur5/ur5.urdf"
        capsule_asset_file = "urdf/magnet_description/capsule.urdf"
        magnet_asset_file = "urdf/magnet_description/magnet.urdf"
        tank1_asset_file = "urdf/magnet_description/tank1.urdf"
        capsule_virtual_asset_file = "urdf/magnet_description/capsule_virtual.urdf"

        # 设置ur5的参数并加载ur5
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True # 适配不同来源的模型
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False # 合并固定关节
        asset_options.disable_gravity = True
        asset_options.use_mesh_materials = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        ur5_asset = self.gym.load_asset(self.sim,asset_root,ur5_asset_file,asset_options)

        # 设置capsule的参数并加载capsule
        asset_options.flip_visual_attachments = True
        asset_options.vhacd_enabled = True # 碰撞形状相关参数
        asset_options.disable_gravity = True
        asset_options.fix_base_link = False
        asset_options.use_mesh_materials = False
        capsule_asset = self.gym.load_asset(self.sim,asset_root,capsule_asset_file,asset_options)
        capsule_props = self.gym.get_asset_rigid_shape_properties(capsule_asset)
        for p in capsule_props:
            p.restitution = 0.6 # 设置恢复系数
        self.gym.set_asset_rigid_shape_properties(capsule_asset, capsule_props)

        # 设置magnet的参数并加载magnet
        asset_options.flip_visual_attachments = True
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True # 合并固定关节
        magnet_asset = self.gym.load_asset(self.sim,asset_root,magnet_asset_file,asset_options)

        # 设置tank1的参数并加载tank1
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        tank1_asset = self.gym.load_asset(self.sim,asset_root,tank1_asset_file,asset_options)

        asset_options.use_mesh_materials = False
        capsule_virtual_asset = self.gym.load_asset(self.sim,asset_root,capsule_virtual_asset_file,asset_options)

        tank1_props = self.gym.get_asset_rigid_shape_properties(tank1_asset)
        for p in tank1_props:
            p.restitution = 1 # 设置恢复系数
        self.gym.set_asset_rigid_shape_properties(tank1_asset, tank1_props)

        # 获取ur5的关节数量
        self.num_ur5_dofs = self.gym.get_asset_dof_count(ur5_asset)

        # 设置ur5的刚度和阻尼
        ur5_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400], dtype=torch.float, device=self.device)
        ur5_dof_damping = to_torch([200, 200, 200, 200, 200, 200], dtype=torch.float, device=self.device)

        # 设置ur5的关节性质
        ur5_dof_props = self.gym.get_asset_dof_properties(ur5_asset)
        self.ur5_dof_lower_limits = []
        self.ur5_dof_upper_limits = []

        # 遍历每个ur5关节,设置关节属性
        for i in range(self.num_ur5_dofs):
            ur5_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            ur5_dof_props['stiffness'][i] = ur5_dof_stiffness[i]
            ur5_dof_props['damping'][i] = ur5_dof_damping[i]

            self.ur5_dof_lower_limits.append(ur5_dof_props['lower'][i])
            self.ur5_dof_upper_limits.append(ur5_dof_props['upper'][i])
        
        self.ur5_dof_lower_limits = to_torch(self.ur5_dof_lower_limits,device=self.device)
        self.ur5_dof_upper_limits = to_torch(self.ur5_dof_upper_limits,device=self.device)

        # 设置ur5的初始位置和姿态
        ur5_start_pose = gymapi.Transform()
        ur5_start_pose.p = gymapi.Vec3(0.0,0.0,0.0)
        ur5_start_pose.r = gymapi.Quat(0.0,0.0,0.0,1.0)

        # 设置capsule的初始位置和姿态
        capsule_start_pose = gymapi.Transform()
        capsule_start_pose.p = gymapi.Vec3(0.5,0.1,0.325)
        capsule_start_pose.r = gymapi.Quat(0.0,1.0,0.0,0.0) # (0,1,0,0)

        # 设置magnet的初始位置和姿态
        magnet_start_pose = gymapi.Transform()
        magnet_start_pose.p = gymapi.Vec3(0.5,0.1,0.575)
        magnet_start_pose.r = gymapi.Quat(0.0,1.0,0.0,0.0)

        # 设置tank1的初始位置和姿态
        tank_start_pose = gymapi.Transform()
        tank_start_pose.p = gymapi.Vec3(0.5,0.1,0.2)
        tank_start_pose.r = gymapi.Quat(0.0000, 0.0, 0.0, 1.0)

        # 设置虚拟capsule的位置和姿态
        capsule_virtual_start_pose = gymapi.Transform()
        capsule_virtual_start_pose.p = gymapi.Vec3(0.5,0.1,0.325)
        capsule_virtual_start_pose.r = gymapi.Quat(0.0,1.0,0.0,0.0)
    

        # 各个句柄
        self.ur5_handles = []
        self.envs = []
        self.capsule_handles = []
        self.default_capsule_states = []
        self.default_magnet_states = []
        self.default_tank_states = []
        self.default_capsule_virtual_states = []
        self.magnet_handles = []
        self.tank1_handles = []

        # 遍历所有环境
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(
                self.sim,lower,upper,num_per_row
            )
            
            # 每个环境中的句柄
            ur5_handle = self.gym.create_actor(env_ptr,ur5_asset,ur5_start_pose,"ur5",i,0,0)
            capsule_handle = self.gym.create_actor(env_ptr,capsule_asset,capsule_start_pose,"capsule",i,0,0)
            magnet_handle = self.gym.create_actor(env_ptr,magnet_asset,magnet_start_pose,"magnet",i,0,0)
            tank1_handle = self.gym.create_actor(env_ptr,tank1_asset,tank_start_pose,"tank1",i,0,0)
            capsule_virtual_handle = self.gym.create_actor(env_ptr,capsule_virtual_asset,capsule_virtual_start_pose,"capsule_virtual",i,0,0)

            # 设置ur5的关节属性
            self.gym.set_actor_dof_properties(env_ptr, ur5_handle, ur5_dof_props)

            self.default_capsule_states.append([capsule_start_pose.p.x, capsule_start_pose.p.y, capsule_start_pose.p.z,
                                    capsule_start_pose.r.x, capsule_start_pose.r.y, capsule_start_pose.r.z, capsule_start_pose.r.w,
                                    0.0,0.0,-0.00001,0.0,0.0,0.0])
            
            self.default_magnet_states.append([magnet_start_pose.p.x,magnet_start_pose.p.y,magnet_start_pose.p.z,
                                               magnet_start_pose.r.x,magnet_start_pose.r.y,magnet_start_pose.r.z,magnet_start_pose.r.w,
                                               0,0,0,0,0,0])
            
            self.default_tank_states.append([tank_start_pose.p.x,tank_start_pose.p.y,tank_start_pose.p.z,
                                               tank_start_pose.r.x,tank_start_pose.r.y,tank_start_pose.r.z,tank_start_pose.r.w,
                                               0,0,0,0,0,0])
            
            self.default_capsule_virtual_states.append([capsule_virtual_start_pose.p.x,capsule_virtual_start_pose.p.y,capsule_virtual_start_pose.p.z,
                                               capsule_virtual_start_pose.r.x,capsule_virtual_start_pose.r.y,capsule_virtual_start_pose.r.z,capsule_virtual_start_pose.r.w,
                                               0,0,0,0,0,0])           

            self.envs.append(env_ptr)
            self.ur5_handles.append(ur5_handle)
            self.capsule_handles.append(capsule_handle)
            self.magnet_handles.append(magnet_handle)
            self.tank1_handles.append(tank1_handle)

        # ur5末端句柄
        self.ur5_ee_handle = self.gym.find_actor_rigid_body_handle(env_ptr,ur5_handle,"ee_link")

        # 获取ur5雅可比矩阵
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur5")
        self.jacobian = gymtorch.wrap_tensor(_jacobian)
        ur5_link_dict = self.gym.get_asset_rigid_body_dict(ur5_asset)
        ur5_ee_index = ur5_link_dict["ee_link"]
        self.j_eef = self.jacobian[:, ur5_ee_index-1, :, :6]

        # 将capsule和magnet的初始位姿转换成tensor
        self.default_capsule_states = to_torch(self.default_capsule_states, device=self.device, dtype=torch.float).view(self.num_envs, 1, 13)
        self.default_magnet_states = to_torch(self.default_magnet_states, device=self.device,dtype=torch.float).view(self.num_envs, 1, 13)
        self.default_tank_states = to_torch(self.default_tank_states, device=self.device,dtype=torch.float).view(self.num_envs, 1, 13)
        self.default_capsule_virtual_states = to_torch(self.default_capsule_virtual_states, device=self.device,dtype=torch.float).view(self.num_envs, 1, 13)

    def compute_reward(self):
        self.rew_buf[:],self.reset_buf[:] = compute_ur5_reward(
            self.reset_buf,self.progress_buf,self.max_episode_length,self.to_target,
            self.to_balance,self.to_target_rot,self.capsule_pos
        )

    def compute_observations(self):
        # 更新状态张量
        self.refresh_tensor()
        
        # 胶囊位置到目标位置的坐标差，胶囊位置到平衡位置的坐标差
        self.capsule_pos = self.capsule_states[:,:,0:3].clone().to(self.device).squeeze()
        self.capsule_rot = self.capsule_states[:,:,3:7].clone().to(self.device).squeeze()
        self.magnet_pos = self.magnet_states[:,:,0:3].squeeze()
        self.magnet_rot = self.magnet_states[:,:,3:7].squeeze()
        balance_point = self.magnet_pos+self.magnet_balance_dis
        self.to_balance = balance_point-self.capsule_pos
        self.to_target = self.target_pos - self.capsule_pos
        self.to_target_rot = quat_diff_rad(self.capsule_rot,self.target_rot).unsqueeze(1)

        # self.obs_buf = torch.cat((self.capsule_pos,self.magnet_pos,self.target_pos,self.ur5_dof_pos,self.to_target),dim=-1)
        self.obs_buf = torch.cat((self.capsule_pos,self.target_pos,
                                  self.capsule_rot,self.target_rot,
                                  self.magnet_pos,self.magnet_rot),dim=-1)

        return self.obs_buf

    def reset_idx(self,env_ids):
        # 设定随机目标点
        self.target_pos[env_ids,] = to_torch([0.5,0.1,0.325],device=self.device)+\
                                (torch.rand((len(env_ids),3),device=self.device)*0.01-0.005)
        
        self.target_rot[env_ids,] = to_torch([0.0, 9.8769e-01, -1.5643e-01, 0.0],device=self.device)
        self.target_capsule_states = torch.cat((self.target_pos,self.target_rot),dim=-1).unsqueeze(1)

        # 环境id的全局索引，此处的切片为ur5的全局索引
        multi_env_ids_int32 = self.global_indices[env_ids, :self.num_props-self.fixed_obj].flatten()

        # 截断函数，设置ur5目标关节角度
        pos = tensor_clamp(
            self.ur5_default_dof_pos.unsqueeze(0),
            self.ur5_dof_lower_limits, self.ur5_dof_upper_limits)
        self.ur5_dof_pos[env_ids, :self.num_ur5_dofs] = pos
        self.ur5_dof_vel[env_ids, :self.num_ur5_dofs] = torch.zeros_like(self.ur5_dof_vel[env_ids])

        # 设置初始ur5关节角度张量      
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        
        # prop_indices为胶囊和磁体的全局索引
        prop_indices = self.global_indices[env_ids, 1:].flatten()

        # 重置根根刚体的位置和姿态
        self.magnet_states[env_ids] = self.default_magnet_states[env_ids]
        self.capsule_states[env_ids,:] = self.default_capsule_states[env_ids,:]
        self.tank1_states[env_ids] = self.default_tank_states[env_ids]
        self.capsule_virtual_states[env_ids,:,0:7] = self.target_capsule_states[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_state_tensor),
                                                    gymtorch.unwrap_tensor(prop_indices), len(prop_indices))
        
        # reset progress_buf和reset_buf
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
            
    def pre_physics_step(self,actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # 到达一定条件进行reset
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            self.gym.simulate(self.sim)
            self.refresh_tensor()

        # actions的范围为（-1，1）
        self.actions = actions.clone().to(self.device)

        self.capsule_vel = self.capsule_states.clone().to(self.device)[:,:,7:13].squeeze()

        # 计算位置和姿态的误差，求解逆运动学
        # pos_err = self.actions*0.01 + self.ur5_ee_pos_goal - self.rigid_body_states[:, self.ur5_ee_handle][:, 0:3]
        # orn_err = orientation_error(self.ur5_ee_rot_goal,self.rigid_body_states[:, self.ur5_ee_handle][:, 3:7])

        pos_err = self.actions[:,0:3]*0.004

        ur5_ee_rot = self.rigid_body_states[:, self.ur5_ee_handle][:, 3:7]
        orn_delta_euler = self.actions[:,3:6]*0.05
        orn_delta_quat = quat_from_euler_xyz(orn_delta_euler[:,0],orn_delta_euler[:,1],orn_delta_euler[:,2])
        orn_desired = quat_mul(ur5_ee_rot,orn_delta_quat)
        orn_err = orientation_error(orn_desired,ur5_ee_rot)

        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        targets = self.ur5_dof_pos+control_ik(dpose.to(self.device),
                                            self.damping,self.j_eef,self.num_envs,self.device)
        pos = tensor_clamp(
            targets,
            self.ur5_dof_lower_limits, self.ur5_dof_upper_limits)
        self.ur5_dof_targets[:, :self.num_ur5_dofs] = pos

        # 设置ur5关节目标角度
        self.gym.set_dof_position_target_tensor(self.sim,gymtorch.unwrap_tensor(self.ur5_dof_targets))
        # self.gym.simulate(self.sim)
        # self.refresh_tensor()

        # 更新并设置磁体的目标位姿
        ur5_ee_pos = self.rigid_body_states[:, self.ur5_ee_handle, 0:3]
        ur5_ee_rot = self.rigid_body_states[:, self.ur5_ee_handle, 3:7]
        magnet_pos,magnet_rot = calculate_magnet_pose(ur5_ee_pos.unsqueeze(1),ur5_ee_rot.unsqueeze(1))
        magnet_vel = torch.zeros((self.num_envs,1,6),device=self.device)
        self.magnet_states[:] = torch.cat((magnet_pos,magnet_rot,magnet_vel),dim=-1)

        self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor))
        # self.gym.simulate(self.sim)
        # self.refresh_tensor()

        # 计算磁矩
        magnet_rot_matrix = quaternion_to_matrix(magnet_rot)
        moment_magnet = self.moment_source_norm*magnet_rot_matrix @ torch.tensor([0.0,0.0,1.0],device=self.device).view(3,1)
        capsule_pos = self.capsule_states.clone().to(self.device)[:,:,0:3]
        capsule_rot = self.capsule_states.clone().to(self.device)[:,:,3:7]
        capsule_rot_matrix = quaternion_to_matrix(capsule_rot)
        moment_capsule = self.moment_capsule_norm*capsule_rot_matrix @ torch.tensor([0.0,0.0,1.0],device=self.device).view(3,1)

        magnetic_force,magnetic_torque = force_moment(capsule_pos-magnet_pos,moment_magnet,moment_capsule,
                                                      device=self.device,num_envs=self.num_envs)

        # 设置水的阻力
        capsule_vel = self.capsule_states.clone().to(self.device)[:,:,7:10].squeeze()
        if torch.norm(capsule_vel,p=2) == 0:
            fluid_drag_force = 0
        else:
            capsule_vel_vector = capsule_vel/torch.norm(capsule_vel,p=2,dim=-1,keepdim=True)
            capsule_vel_norm = torch.norm(capsule_vel,p=2,dim=-1,keepdim=True)
            fluid_drag_force = -0.5*1000*0.4*0.001256*capsule_vel_norm**2*capsule_vel_vector
            
        # 设置磁力和磁力矩
        force_tensor = torch.zeros(self.num_envs*15,3,device=self.device)
        torque_tensor = torch.zeros(self.num_envs*15,3,device=self.device)
        torque_tensor[11: :15,:] = magnetic_torque.squeeze(1)
        force_tensor[11: :15,:] = magnetic_force.squeeze(1)+self.net_weight.unsqueeze(0)+fluid_drag_force

        self.gym.apply_rigid_body_force_tensors(self.sim,gymtorch.unwrap_tensor(force_tensor), 
                                                gymtorch.unwrap_tensor(torque_tensor), gymapi.ENV_SPACE)
        

    def post_physics_step(self):
        # 步数更新
        self.progress_buf += 1

        # 计算状态和奖励
        self.compute_observations()
        self.compute_reward()

        # if self.viewer and self.debug_viz:
        #     self.gym.clear_lines(self.viewer)
        #     for i in range(self.num_envs):
        #         p_start = self.target_pos[i].cpu().numpy()
        #         p_end_x = to_torch([p_start[0]+0.1,p_start[1],p_start[2]],device=self.device).cpu().numpy()
        #         p_end_y = to_torch([p_start[0],p_start[1]+0.1,p_start[2]],device=self.device).cpu().numpy()
        #         p_end_z = to_torch([p_start[0],p_start[1],p_start[2]+0.1],device=self.device).cpu().numpy()
        #         self.gym.add_lines(self.viewer,self.envs[i],1,[p_start[0],p_start[1],p_start[2],p_end_x[0],p_end_x[1],p_end_x[2]],[1.0, 0.0, 0.0])
        #         self.gym.add_lines(self.viewer,self.envs[i],1,[p_start[0],p_start[1],p_start[2],p_end_y[0],p_end_y[1],p_end_y[2]],[0.0, 1.0, 0.0])
        #         self.gym.add_lines(self.viewer,self.envs[i],1,[p_start[0],p_start[1],p_start[2],p_end_z[0],p_end_z[1],p_end_z[2]],[0.0, 0.0, 1.0])

    def refresh_tensor(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

@torch.jit.script
def compute_ur5_reward(reset_buf,progress_buf,max_episode_length,to_target,
                       to_balance,to_target_rot,capsule_pos):
    # type: (Tensor, Tensor, float, Tensor,Tensor,Tensor,Tensor) -> Tuple[Tensor, Tensor]

    d = torch.norm(to_target, p=2, dim=-1)
    
    # 到达平衡点的奖励和到达目标点的奖励
    dist_to_target_reward = torch.exp(-d*50*10)-0.6
    # print(dist_to_target_reward)
    rot_to_target_reward = torch.exp(-abs(to_target_rot.squeeze())/torch.pi*10)

    # 总奖励
    rewards = dist_to_target_reward + rot_to_target_reward
    # print(capsule_pos[63,:])

    rewards = torch.where((abs(capsule_pos[:,0]-0.5)>0.035) |
                          (abs(capsule_pos[:,1]-0.1)>0.035) |
                          (abs(capsule_pos[:,2]-0.325)>0.035),rewards-100,rewards)

    reset_buf = torch.where(progress_buf >= max_episode_length,torch.ones_like(reset_buf),reset_buf)

    reset_buf = torch.where(abs(capsule_pos[:,0]-0.5)>0.035,torch.ones_like(reset_buf),reset_buf)
    reset_buf = torch.where(abs(capsule_pos[:,1]-0.1)>0.035,torch.ones_like(reset_buf),reset_buf)
    reset_buf = torch.where(abs(capsule_pos[:,2]-0.325)>0.035,torch.ones_like(reset_buf),reset_buf)


    return rewards,reset_buf

@torch.jit.script
def control_ik(dpose, damping, j_eef, num_envs,device):
    # type: (Tensor, float, Tensor, int, str) -> Tensor

    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 6)
    # print(j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda)@ dpose)
    return u

@torch.jit.script
def orientation_error(desired, current):
    # type: (Tensor,Tensor) -> Tensor
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

@torch.jit.script
def quat_to_rot_matrix(quat):
    # type:(Tensor) -> Tensor
    x,y,z,w = quat[:,0],quat[:,1],quat[:,2],quat[:,3]
    rot_matrix = torch.stack([
        1 - 2 * (y**2 + z**2),  2 * (x * y - z * w),  2 * (x * z + y * w),
        2 * (x * y + z * w),    1 - 2 * (x**2 + z**2),  2 * (y * z - x * w),
        2 * (x * z - y * w),    2 * (y * z + x * w),    1 - 2 * (x**2 + y**2)
    ], dim=1).view(-1, 3, 3) 
    return rot_matrix

@torch.jit.script
def calculate_magnet_pose(ur5_ee_pos,ur5_ee_rot):
    # type: (Tensor,Tensor) -> Tuple[Tensor,Tensor]
    ur5_ee_pos = ur5_ee_pos.squeeze()
    ur5_ee_rot = ur5_ee_rot.squeeze()
    quat_rot_euler = torch.tensor([0.0,torch.pi/2,0.0],device="cuda:0").unsqueeze(0)+torch.zeros_like(ur5_ee_pos)
    quat_rot = quat_from_euler_xyz(quat_rot_euler[:,0],quat_rot_euler[:,1],quat_rot_euler[:,2])
    # magnet_rot_quat = ur5_ee_rot
    magnet_rot_quat = quat_mul(ur5_ee_rot,quat_rot)
    magnet_rot_matrix = quat_to_rot_matrix(magnet_rot_quat)
    # magnet_rot_matrix = quat_to_rot_matrix(ur5_ee_rot)
    magnet_pos = ur5_ee_pos + magnet_rot_matrix[:,2,0:3]*0.025 # 64*3*3
    magnet_pos = magnet_pos.unsqueeze(1)
    magnet_rot_quat = magnet_rot_quat.unsqueeze(1)
    return magnet_pos,magnet_rot_quat

@torch.jit.script
def force_moment(p, ma, mc, device, num_envs):
    """
    计算永磁铁在空间中某位置对被驱动永磁铁产生的磁力和磁力矩
    :param pa: 永磁铁的位置 array 3*1
    :param pc: 空间中被驱动永磁铁的位置 array 3*1
    :param ma: 永磁铁的磁矩 array 3*1
    :param mc: 被驱动磁铁的磁矩 array 3*1
    :return: 磁力 force 3*1 和 磁力矩 moment 3*1
    """
    # type: (Tensor,Tensor,Tensor,str,int)->Tuple[Tensor,Tensor]

    #输入为num_envs*1*3
    k = 4 * torch.pi * 1e-7
    ma = ma.view(num_envs,-1,1)
    mc = mc.view(num_envs,-1,1)
    p = p.view(num_envs,-1,1)
    p_norm = torch.norm(p,dim=1,keepdim=True)
    p_hat = p/p_norm
    p_trans = torch.transpose(p,1,2)
    p_hat_trans = torch.transpose(p_hat,1,2)
    ma_trans = torch.transpose(ma,1,2)
    eye_3 = torch.eye(3,device=device).unsqueeze(0).expand(num_envs,3,3)
    
    # 磁源在某点产生的磁场强度
    field = k / (4*torch.pi*pow(p_norm,5)) * ((3*(p@p_trans)-pow(p_norm,2) * eye_3) @ ma)

    # 磁源在某点产生的磁场梯度矩阵
    gradient = 3 * k / (4*torch.pi*pow(p_norm,4))*(ma@p_hat_trans+p_hat@ma_trans+(p_hat_trans@ma)*(eye_3-5*(p_hat@p_hat_trans)))
    S_mc = torch.tensor([[0.0, float(-mc[0,2, 0]), float(mc[0,1, 0])],
                   [float(mc[0,2, 0]), 0.0, float(-mc[0,0, 0])],
                   [float(-mc[0,1, 0]), float(mc[0,0, 0]), 0.0]],device=device)
    torch.tensor(S_mc.unsqueeze(0).expand(num_envs,3,3))
    force = torch.matmul(gradient,mc).view(num_envs,1,3)
    moment = torch.matmul(S_mc,field).view(num_envs,1,3)
    
    return force,moment
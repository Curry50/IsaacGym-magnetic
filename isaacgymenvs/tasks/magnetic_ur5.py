import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import to_torch, quaternion_to_matrix, tensor_clamp,quat_diff_rad
from isaacgym.torch_utils import quat_conjugate,quat_mul,get_euler_xyz,quat_from_euler_xyz
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MagneticUr5(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.damping = 1
        self.max_episode_length = 900 # 600

        self.cfg["env"]["numObservations"] = 21
        self.cfg["env"]["numActions"] = 3

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

        # 单个环境里actor的总数量以及刚体的数量
        self.num_props = 5 
        self.fixed_obj = 4 

        # 全局索引
        self.global_indices = torch.arange(self.num_envs * self.num_props, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        # 由于以下物体是刚体没有关节，需单独处理
        self.capsule_states = self.root_state_tensor[:, self.num_props-self.fixed_obj:self.num_props-3].to(torch.float32)
        self.magnet_states = self.root_state_tensor[:, self.num_props-3:self.num_props-2].to(torch.float32)
        self.tank1_states = self.root_state_tensor[:, self.num_props-2:self.num_props-1].to(torch.float32)
        self.capsule_virtual_states = self.root_state_tensor[:,self.num_props-1:].to(torch.float32)

        # 初始化相关变量
        self.data_initialization()

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # reset/初始化
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def data_initialization(self):
        # 初始化ur5关节目标角度张量，ur5初始角度
        self.ur5_dof_targets = torch.zeros((self.num_envs,self.num_ur5_dofs),dtype=torch.float32,device=self.device)
        self.ur5_default_dof_pos = to_torch([-0.0317,  0.0819,  1.8445, -1.9600, -1.6012, -0.0198], device=self.device)

        # 设置ur5的目标位置和姿态
        self.ur5_ee_pos_goal = torch.zeros_like(self.rigid_body_states[:,self.ur5_ee_handle][:, 0:3])
        self.ur5_ee_rot_goal = torch.zeros_like(self.rigid_body_states[:,self.ur5_ee_handle][:, 3:7])
        self.ur5_ee_pos_goal[:,] = to_torch([0.5,0.1,0.3],device=self.device)
        self.ur5_ee_rot_goal[:,] = to_torch([0.0000, 0.7071, 0.0000, 0.7071],device=self.device)
        
        # 初始化画图所需的变量
        self.time_steps = []
        self.position_errors = []
        self.capsule_pos_cpu_list = []
        self.path_count = torch.zeros((self.num_envs,1),device=self.device,dtype=torch.int)

        # 初始化胶囊的目标位置,胶囊的初始位置和平衡位置
        self.target_pos = torch.zeros((self.num_envs,3),device=self.device)
        self.target_rot = torch.zeros((self.num_envs,4),device=self.device)
        self.capsule_start_pos = torch.tensor([0.5,0.1,0.1126],device=self.device)
        self.magnet_start_pos = torch.tensor([0.5,0.1,0.175],device=self.device)
        self.capsule_start_ori_euler = torch.zeros((self.num_envs,3),device=self.device)
        self.capsule_start_ori_euler[:,0] = torch.pi
        self.capsule_start_ori_euler[:,1] = torch.pi/2
        self.capsule_start_ori_euler[:,2] = 0.0
        self.capsule_pos = torch.zeros((self.num_envs,3),device=self.device)


        # 设置磁体和胶囊的磁矩大小
        self.moment_source_norm = 26.2 # 26.2
        self.moment_capsule_norm = 0.126

        # 初始化相关变量
        self.last_capsule_states = torch.zeros_like(self.capsule_states,device=self.device)
        self.last_capsule_virtual_states = torch.zeros_like(self.capsule_virtual_states,device=self.device)
        self.to_target = torch.zeros((self.num_envs,3),device=self.device)
        self.to_target_rot = torch.zeros((self.num_envs,1),device=self.device)
        self.ur5_last_dof_pos = torch.zeros_like(self.ur5_dof_pos,device=self.device)
        self.last_capsule_states = self.capsule_states.clone().to(self.device)
        self.ur5_last_dof_pos = self.ur5_dof_pos.clone().to(self.device)
        self.last_capsule_virtual_states = self.capsule_virtual_states.clone().to(self.device)

        # 设置目标点与胶囊的距离
        self.target_dis = 0.01

        # 设置工作空间的大小
        self.ws_length = 0.03

        # 当达到以下阈值时，胶囊到达目标位姿
        self.d_min = 0.0025
        self.theta_min = torch.pi/180

        # 缩放尺度
        self.action_trans_scale = 0.02
        self.action_rot_scale = 0.1
        self.drag_force_scale = 4e-2
        self.drag_torque_scale = 5e-6

        # 测试机械臂平动和转动带来的效果
        self.translation = torch.zeros((self.num_envs,3),device=self.device)
        self.translation[:,1] = -0.002*0
        self.translation[:,0] = -0.006*0
        self.rotation = torch.zeros((self.num_envs,3),device=self.device)
        self.rotation[:,2] = 0.1

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
        asset_options.disable_gravity = False
        asset_options.fix_base_link = False
        asset_options.use_mesh_materials = False
        capsule_asset = self.gym.load_asset(self.sim,asset_root,capsule_asset_file,asset_options)
        capsule_props = self.gym.get_asset_rigid_shape_properties(capsule_asset)
        
        for p in capsule_props:
            p.restitution = 0.0 # 设置恢复系数
            # p.rolling_friction = 0.00
            p.friction = 0.005
            # p.torsion_friction = 0.0001
        self.gym.set_asset_rigid_shape_properties(capsule_asset, capsule_props)

        # 设置magnet的参数并加载magnet
        asset_options.flip_visual_attachments = True
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True # 合并固定关节
        magnet_asset = self.gym.load_asset(self.sim,asset_root,magnet_asset_file,asset_options)

        # 设置tank1的参数并加载tank1
        asset_options.flip_visual_attachments = True
        asset_options.vhacd_enabled = True # 碰撞形状相关参数
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.use_mesh_materials = False
        tank1_asset = self.gym.load_asset(self.sim,asset_root,tank1_asset_file,asset_options)

        asset_options.use_mesh_materials = False
        capsule_virtual_asset = self.gym.load_asset(self.sim,asset_root,capsule_virtual_asset_file,asset_options)

        tank1_props = self.gym.get_asset_rigid_shape_properties(tank1_asset)
        for p in tank1_props:
            p.restitution = 0.0 # 设置恢复系数
            # p.rolling_friction = 0.000
            p.friction = 0.005
            # p.torsion_friction = 0.0001
        self.gym.set_asset_rigid_shape_properties(tank1_asset, tank1_props)

        # 获取ur5的关节数量
        self.num_ur5_dofs = self.gym.get_asset_dof_count(ur5_asset)

        # 设置ur5的刚度和阻尼
        ur5_dof_stiffness = to_torch([300, 300, 300, 300, 300, 300], dtype=torch.float32, device=self.device)
        ur5_dof_damping = to_torch([100, 100, 100, 100, 100, 100], dtype=torch.float32, device=self.device)

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
        capsule_start_pose.p = gymapi.Vec3(0.5,0.1,0.1126)
        capsule_start_pose.r = gymapi.Quat(7.0711e-01, -3.0909e-08, -7.0711e-01, -3.0909e-08) # (0,1,0,0)

        # 设置magnet的初始位置和姿态
        magnet_start_pose = gymapi.Transform()
        magnet_start_pose.p = gymapi.Vec3(0.5,0.1,0.575)
        magnet_start_pose.r = gymapi.Quat(0,1.0,0.0,0.0)

        # 设置tank1的初始位置和姿态
        tank_start_pose = gymapi.Transform()
        tank_start_pose.p = gymapi.Vec3(0.5,0.1,0.1)
        tank_start_pose.r = gymapi.Quat(0.0000, 0.0, 0.0, 1.0)

        # 设置虚拟capsule的位置和姿态
        capsule_virtual_start_pose = gymapi.Transform()
        capsule_virtual_start_pose.p = gymapi.Vec3(0.5,0.1,0.1126)
        capsule_virtual_start_pose.r = gymapi.Quat(7.0711e-01, -3.0909e-08, -7.0711e-01, -3.0909e-08)
    

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

            self.gym.set_actor_scale(env_ptr,tank1_handle,0.6)

            # 设置ur5的关节属性
            self.gym.set_actor_dof_properties(env_ptr, ur5_handle, ur5_dof_props)

            self.default_capsule_states.append([capsule_start_pose.p.x, capsule_start_pose.p.y, capsule_start_pose.p.z,
                                    capsule_start_pose.r.x, capsule_start_pose.r.y, capsule_start_pose.r.z, capsule_start_pose.r.w,
                                    0.0000001,0,0,0,0,0])
            
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
        self.ur5_ee_index = ur5_link_dict["ee_link"]
        self.j_eef = self.jacobian[:, self.ur5_ee_index-1, :, :6]

        # 将capsule和magnet的初始位姿转换成tensor
        self.default_capsule_states = to_torch(self.default_capsule_states, device=self.device, dtype=torch.float32).view(self.num_envs, 1, 13)
        self.default_magnet_states = to_torch(self.default_magnet_states, device=self.device,dtype=torch.float32).view(self.num_envs, 1, 13)
        self.default_tank_states = to_torch(self.default_tank_states, device=self.device,dtype=torch.float32).view(self.num_envs, 1, 13)
        self.default_capsule_virtual_states = to_torch(self.default_capsule_virtual_states, device=self.device,dtype=torch.float32).view(self.num_envs, 1, 13)

    def compute_reward(self,actions):
        self.rew_buf[:],self.reset_buf[:] = compute_ur5_reward(
            self.reset_buf,self.progress_buf,self.max_episode_length,self.to_target,
            self.capsule_pos,self.to_target_rot,self.delta_x_angle,
            self.d_min,self.theta_min,self.capsule_start_pos,self.ws_length
        )

    def compute_observations(self):
        # 更新状态张量
        self.refresh_tensor()
        
        # 胶囊位置到目标位置的坐标差，胶囊位置到平衡位置的坐标差
        self.capsule_pos = self.capsule_states[:,:,0:3].clone().to(self.device).squeeze()
        self.capsule_rot = self.capsule_states[:,:,3:7].clone().to(self.device).squeeze()
        self.magnet_pos = self.magnet_states[:,:,0:3].squeeze()
        self.magnet_rot = self.magnet_states[:,:,3:7].squeeze()     
        self.to_target = self.target_pos - self.capsule_pos
        self.to_target_rot = quat_diff_rad(self.capsule_rot,self.target_rot).unsqueeze(1)

        capsule_rot_matrix = quaternion_to_matrix(self.capsule_rot).view(self.num_envs,9)
        target_rot_matrix = quaternion_to_matrix(self.target_rot).view(self.num_envs,9)
        magnet_rot_matrix = quaternion_to_matrix(self.magnet_rot).view(self.num_envs,9)

        self.obs_buf = torch.cat((self.capsule_rot,self.target_rot,self.magnet_rot,
                                  self.capsule_pos,self.target_pos,self.magnet_pos),dim=-1)


        return self.obs_buf

    def reset_idx(self,env_ids):

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

        # 计算方向向量
        random_directions = torch.randn((self.num_envs,3),device=self.device)
        random_directions[:,2] = torch.zeros((self.num_envs),device=self.device)
        norms = torch.norm(random_directions,dim=1,p=2,keepdim=True)        
        unit_directions = random_directions/norms

        # 随机生成目标姿态
        a = torch.rand((self.num_envs,1),device=self.device)
        self.delta_x_angle = a * torch.pi/3 - torch.pi/6
        self.capsule_start_ori_euler_clone = self.capsule_start_ori_euler.clone().to(self.device)
        self.capsule_start_ori_euler_clone[:,0] = self.capsule_start_ori_euler_clone[:,0] + self.delta_x_angle[:,0]
        self.target_ori = quat_from_euler_xyz(self.capsule_start_ori_euler_clone[:,0],self.capsule_start_ori_euler_clone[:,1],self.capsule_start_ori_euler_clone[:,2])


        # 设定随机目标点
        self.target_pos[env_ids,] = self.capsule_start_pos \
            + unit_directions[env_ids] * self.target_dis                         
        
        self.target_rot[env_ids] = self.target_ori[env_ids]
        self.target_capsule_states = torch.cat((self.target_pos,self.target_rot),dim=-1).unsqueeze(1).to(torch.float32)
        
        # prop_indices为胶囊和磁体的全局索引
        prop_indices = self.global_indices[env_ids, 1:].flatten()

        # 重置根根刚体的位置和姿态
        self.magnet_states[env_ids] = self.default_magnet_states[env_ids]
        self.capsule_states[env_ids] = self.default_capsule_states[env_ids]
        self.tank1_states[env_ids] = self.default_tank_states[env_ids]
        self.capsule_virtual_states[env_ids,:,0:7] = self.target_capsule_states[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_state_tensor),
                                                    gymtorch.unwrap_tensor(prop_indices), len(prop_indices))
        
        # reset progress_buf和reset_buf
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    # 更新目标点
    def reset_random_target(self,env_ids):
        prop_indices = self.global_indices[env_ids, 1:].flatten()

        # 计算方向向量
        random_directions = torch.randn((self.num_envs,3),device=self.device)
        random_directions[:,2] = torch.zeros((self.num_envs),device=self.device)
        norms = torch.norm(random_directions,dim=1,p=2,keepdim=True)        
        unit_directions = random_directions/norms

        # 随机生成目标姿态
        a = torch.rand((self.num_envs,1),device=self.device)
        capsule_rot = torch.zeros((self.num_envs,3),device=self.device)
        capsule_rot[:,0],capsule_rot[:,1],capsule_rot[:,2] = get_euler_xyz(self.capsule_virtual_states[:,:,3:7].clone().to(self.device).squeeze())
        self.delta_x_angle = a * torch.pi/4 + torch.pi/4
        capsule_rot[:,0] = capsule_rot[:,0] + self.delta_x_angle[:,0]*0
        self.target_ori = quat_from_euler_xyz(capsule_rot[:,0],capsule_rot[:,1],capsule_rot[:,2])
        
        # 更新目标
        self.capsule_pos = self.capsule_virtual_states[:,:,0:3].clone().to(self.device).squeeze()
        self.target_pos[env_ids] = self.capsule_pos[env_ids] + unit_directions[env_ids] * self.target_dis
        self.target_rot[env_ids] = self.target_ori[env_ids]
        self.target_capsule_states = torch.cat((self.target_pos,self.target_rot),dim=-1).unsqueeze(1)
        self.capsule_virtual_states[env_ids,:,0:7] = self.target_capsule_states[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.root_state_tensor),
                                            gymtorch.unwrap_tensor(prop_indices), len(prop_indices))

        self.reset_buffer(env_ids) 

    def reset_buffer(self,env_ids):
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
    
    def refresh_tensor(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
            
    def pre_physics_step(self,actions):
        # 超出工作空间的标志符
        self.reset_ws_buf = torch.where((abs(self.capsule_pos[:,0]-self.capsule_start_pos[0])>self.ws_length) |
                        (abs(self.capsule_pos[:,1]-self.capsule_start_pos[1])>self.ws_length) |
                        (abs(self.capsule_pos[:,2]-self.capsule_start_pos[2])>self.ws_length),torch.ones_like(self.reset_buf),torch.zeros_like(self.reset_buf))
        
        # 超出最大步数的标志符``
        d = torch.norm(self.to_target, p=2, dim=-1)
        self.theta_arrived = abs(self.to_target_rot.squeeze()) <= self.theta_min
        # self.reset_el_buf: torch.Tensor = torch.where((self.progress_buf >= self.max_episode_length)&((d > self.d_min)|(~self.theta_arrived)),torch.ones_like(self.reset_buf),torch.zeros_like(self.reset_buf))
        self.reset_el_buf = torch.where(self.progress_buf >= self.max_episode_length,torch.ones_like(self.reset_buf),torch.zeros_like(self.reset_buf))
        # 达到目标位姿的标志符
        self.reset_d_buf = torch.where((d <= self.d_min)&(self.progress_buf >= self.max_episode_length)&(self.theta_arrived),torch.ones_like(self.reset_buf),torch.zeros_like(self.reset_buf))

        # 超出工作空间的环境id，大于最大步数的环境id，达到目标的环境id
        env_ids_ws = self.reset_ws_buf.nonzero(as_tuple=False).flatten()
        env_ids_el = self.reset_el_buf.nonzero(as_tuple=False).flatten()
        env_ids_d = self.reset_d_buf.nonzero(as_tuple=False).flatten()

        # 如果超出工作空间，重置环境
        # if len(env_ids_ws) > 0:
        #     self.reset_idx(env_ids_ws)  

        # 如果大于最大步数
        if len(env_ids_el) > 0:
            # 更新目标和buffer
            self.reset_idx(env_ids_el)

        # if len(env_ids_d) > 0:
        #     # self.reset_random_target(env_ids_d)
        #     self.reset_idx(env_ids_d)

        # actions的范围为（-1，1）
        self.actions = actions.clone().to(self.device)

        self.capsule_vel = self.capsule_states.clone().to(self.device)[:,:,7:10].squeeze()
        self.capsule_rot_vel = self.capsule_states.clone().to(self.device)[:,:,10:13].squeeze()

        pos_err = self.ur5_ee_pos_goal - self.rigid_body_states[:, self.ur5_ee_handle][:, 0:3]
        orn_err_euler = torch.zeros((self.num_envs,3),device=self.device)
        orn_euler = torch.zeros((self.num_envs,3),device=self.device)
        orn_euler[:,0],orn_euler[:,1],orn_euler[:,2] = get_euler_xyz(self.rigid_body_states[:, self.ur5_ee_handle, 3:7].squeeze())
        orn_err_euler[:,2] = self.actions[:,2]*self.action_rot_scale

        self.ur5_ee_rot = self.rigid_body_states[:, self.ur5_ee_handle, 3:7]
        orn_desired = self.ur5_ee_rot_goal
        # orn_desired_euler = torch.zeros((self.num_envs,3),device=self.device)
        # orn_desired_euler[:,0],orn_desired_euler[:,1],orn_desired_euler[:,2] = get_euler_xyz(orn_desired)
        # orn_desired_euler[:,2] = self.actions[:,0]*self.action_rot_scale
        # orn_desired = quat_from_euler_xyz(orn_desired_euler[:,0],orn_desired_euler[:,1],orn_desired_euler[:,2])
        orn_err = orientation_error(orn_desired,self.ur5_ee_rot) 

        # 位置误差和姿态误差，计算逆运动学
        pos_err[:,0:2] = self.actions[:,0:2]*self.action_trans_scale
        dpose = torch.cat([pos_err, orn_err_euler], -1).unsqueeze(-1)
        targets = self.ur5_dof_pos+control_ik(dpose.to(self.device),
                                            self.damping,self.j_eef,self.num_envs,self.device)
        
        pos = tensor_clamp(
            targets,
            self.ur5_dof_lower_limits, self.ur5_dof_upper_limits)
        self.ur5_dof_targets[:, :self.num_ur5_dofs] = pos

        # 设置ur5关节目标角度
        self.gym.set_dof_position_target_tensor(self.sim,gymtorch.unwrap_tensor(self.ur5_dof_targets))

        # 更新并设置磁体的目标位姿
        self.ur5_ee_pos = self.rigid_body_states[:, self.ur5_ee_handle, 0:3]
        self.ur5_ee_rot = self.rigid_body_states[:, self.ur5_ee_handle, 3:7]
        magnet_pos,magnet_rot = calculate_magnet_pose(self.ur5_ee_pos.unsqueeze(1),self.ur5_ee_rot.unsqueeze(1))
        magnet_vel = torch.zeros((self.num_envs,1,6),device=self.device)
        self.magnet_states[:] = torch.cat((magnet_pos,magnet_rot,magnet_vel),dim=-1)

        self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor))

        # 计算磁矩
        magnet_rot_matrix = quaternion_to_matrix(magnet_rot)
        moment_magnet = self.moment_source_norm*magnet_rot_matrix @ torch.tensor([0.0,0.0,1.0],device=self.device).view(3,1).to(torch.float32)
        capsule_pos = self.capsule_states.clone().to(self.device)[:,:,0:3]
        capsule_rot = self.capsule_states.clone().to(self.device)[:,:,3:7]
        capsule_rot_matrix = quaternion_to_matrix(capsule_rot)
        moment_capsule = self.moment_capsule_norm*capsule_rot_matrix @ torch.tensor([0.0,0.0,1.0],device=self.device).view(3,1).to(torch.float32)

        magnetic_force,magnetic_torque = force_moment(capsule_pos-magnet_pos,moment_magnet,moment_capsule,
                                                      device=self.device,num_envs=self.num_envs)

        force_tensor = torch.zeros(self.num_envs,15,3,device=self.device,dtype=torch.float32)
        torque_tensor = torch.zeros(self.num_envs,15,3,device=self.device,dtype=torch.float32)
        force_tensor[:,11,:] = magnetic_force.squeeze(1)
        torque_tensor[:,11,:] = magnetic_torque.squeeze(1)
        self.gym.apply_rigid_body_force_tensors(self.sim,gymtorch.unwrap_tensor(force_tensor), 
                                                gymtorch.unwrap_tensor(torque_tensor), gymapi.ENV_SPACE)


    def post_physics_step(self):
        ''' 
        以reset_buf作为结束的标志位,如果重置reset_buf,则episode_length无法更新
        如果不重置reset_buf,则整个环境会被reset_idx一起更新(因为共用了reset_buf)
        '''

        # 步数更新
        self.progress_buf += 1

        # 计算状态和奖励
        self.compute_observations()
        self.compute_reward(self.actions)

        # 计算磁矩
        magnet_pos = self.magnet_states.clone().to(self.device)[:,:,0:3]
        magnet_rot = self.magnet_states.clone().to(self.device)[:,:,3:7]

        magnet_rot_matrix = quaternion_to_matrix(magnet_rot)
        moment_magnet = self.moment_source_norm*magnet_rot_matrix @ torch.tensor([0.0,0.0,1.0],device=self.device).view(3,1).to(torch.float32)
        capsule_pos = self.capsule_states.clone().to(self.device)[:,:,0:3]
        capsule_rot = self.capsule_states.clone().to(self.device)[:,:,3:7]
        capsule_rot_matrix = quaternion_to_matrix(capsule_rot)
        moment_capsule = self.moment_capsule_norm*capsule_rot_matrix @ torch.tensor([0.0,0.0,1.0],device=self.device).view(3,1).to(torch.float32)

        magnetic_force,magnetic_torque = force_moment(capsule_pos-magnet_pos,moment_magnet,moment_capsule,
                                                      device=self.device,num_envs=self.num_envs)

        force_tensor = torch.zeros(self.num_envs,15,3,device=self.device,dtype=torch.float32)
        torque_tensor = torch.zeros(self.num_envs,15,3,device=self.device,dtype=torch.float32)
        force_tensor[:,11,:] = magnetic_force.squeeze(1)
        torque_tensor[:,11,:] = magnetic_torque.squeeze(1)
        self.gym.apply_rigid_body_force_tensors(self.sim,gymtorch.unwrap_tensor(force_tensor), 
                                                gymtorch.unwrap_tensor(torque_tensor), gymapi.ENV_SPACE)


@torch.jit.script
def compute_ur5_reward(reset_buf,progress_buf,max_episode_length,to_target,
                       capsule_pos,to_target_rot,delta_x_angle,
                       d_min,theta_min,capsule_start_pos,ws_length):
    # type: (Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, float, float, Tensor, float) -> Tuple[Tensor, Tensor]

    dist_reward_scale = 100
    rot_reawrd_scale = 3

    d = torch.norm(to_target, p=2, dim=-1)

    d_arrived = d <= d_min
    theta_arrived = abs(to_target_rot.squeeze()) <= theta_min

    # dist_to_target_reward = 1/(1+dist_reward_scale*d*d)
    # dist_to_target_reward = torch.exp(-dist_reward_scale*d)
    dist_to_target_reward = -d*50
    rot_to_target_reward = torch.exp(-rot_reawrd_scale*abs(to_target_rot.squeeze()))

    # print(dist_to_target_reward[0])
    # print(rot_to_target_reward[0])

    # 总奖励
    rewards = dist_to_target_reward + rot_to_target_reward

    rewards = torch.where((abs(capsule_pos[:,0]-capsule_start_pos[0])>ws_length) | # 0.01 0.015
                          (abs(capsule_pos[:,1]-capsule_start_pos[1])>ws_length) |
                          (abs(capsule_pos[:,2]-capsule_start_pos[2])>ws_length),rewards-80,rewards)
    
    # rewards = torch.where((d <= d_min)&(progress_buf >= max_episode_length)&(theta_arrived),rewards+120,rewards)

    # 包含max_episode_length和工作空间的reset信息
    # reset_buf = torch.where((progress_buf >= max_episode_length)&((d > d_min)|(~theta_arrived)),torch.ones_like(reset_buf),reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length,torch.ones_like(reset_buf),reset_buf)

    # reset_buf = torch.where((d <= d_min)&(progress_buf >= max_episode_length)&(theta_arrived),torch.ones_like(reset_buf),reset_buf)

    reset_buf = torch.where(abs(capsule_pos[:,0]-capsule_start_pos[0])>ws_length,torch.ones_like(reset_buf),reset_buf)
    reset_buf = torch.where(abs(capsule_pos[:,1]-capsule_start_pos[1])>ws_length,torch.ones_like(reset_buf),reset_buf)
    reset_buf = torch.where(abs(capsule_pos[:,2]-capsule_start_pos[2])>ws_length,torch.ones_like(reset_buf),reset_buf)

    return rewards,reset_buf

@torch.jit.script
def control_ik(dpose, damping, j_eef, num_envs,device):
    # type: (Tensor, float, Tensor, int, str) -> Tensor

    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 6)
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
    quat_rot_euler = torch.tensor([0.0,0.0,0.0],device="cuda:0").unsqueeze(0)+torch.zeros_like(ur5_ee_pos)
    quat_rot = quat_from_euler_xyz(quat_rot_euler[:,0],quat_rot_euler[:,1],quat_rot_euler[:,2])
    # magnet_rot_quat = ur5_ee_rot
    magnet_rot_quat = quat_mul(ur5_ee_rot,quat_rot)
    magnet_rot_matrix = quat_to_rot_matrix(magnet_rot_quat)
    # magnet_rot_matrix = quat_to_rot_matrix(ur5_ee_rot)
    magnet_pos = ur5_ee_pos + magnet_rot_matrix[:,0:3,0]*0.025 # 64*3*3
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
    ma = ma.view(num_envs,-1,1).to(torch.float64)
    mc = mc.view(num_envs,-1,1).to(torch.float64)
    p = p.view(num_envs,-1,1).to(torch.float64)
    p_norm = torch.norm(p,dim=1,keepdim=True)
    p_hat = p/p_norm
    p_trans = torch.transpose(p,1,2).to(torch.float64)
    p_hat_trans = torch.transpose(p_hat,1,2).to(torch.float64)
    ma_trans = torch.transpose(ma,1,2).to(torch.float64)
    eye_3 = torch.eye(3,device=device).unsqueeze(0).expand(num_envs,3,3).to(torch.float64)
    
    # 磁源在某点产生的磁场强度
    field = k / (4*torch.pi*pow(p_norm,5)) * ((3*(p@p_trans)-pow(p_norm,2) * eye_3) @ ma)

    # 磁源在某点产生的磁场梯度矩阵
    gradient = 3 * k / (4*torch.pi*pow(p_norm,4))*(ma@p_hat_trans+p_hat@ma_trans+(p_hat_trans@ma)*(eye_3-5*(p_hat@p_hat_trans)))

    x = mc[:, 0, 0]  # (n,)
    y = mc[:, 1, 0]  # (n,)
    z = mc[:, 2, 0]  # (n,)

    zeros = torch.zeros_like(x)

    S_mc = torch.stack([
        torch.stack([zeros, -z, y], dim=1),
        torch.stack([z, zeros, -x], dim=1),
        torch.stack([-y, x, zeros], dim=1)
        ], dim=1)  # (n, 3, 3)
    force = torch.matmul(gradient,mc).view(num_envs,1,3)
    moment = torch.matmul(S_mc,field).view(num_envs,1,3)

    # print(S_mc)
    
    return force,moment

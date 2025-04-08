from isaacgym.torch_utils import quat_conjugate,quat_mul,get_euler_xyz,quat_from_euler_xyz
from isaacgymenvs.utils.torch_jit_utils import to_torch
import numpy as np
import torch

def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)
def quat_to_rot_matrix(quat):
    x,y,z,w = quat[:,0],quat[:,1],quat[:,2],quat[:,3]
    rot_matrix = torch.stack([
        1 - 2 * (y**2 + z**2),  2 * (x * y - z * w),  2 * (x * z + y * w),
        2 * (x * y + z * w),    1 - 2 * (x**2 + z**2),  2 * (y * z - x * w),
        2 * (x * z - y * w),    2 * (y * z + x * w),    1 - 2 * (x**2 + y**2)
    ], dim=1).view(-1, 3, 3) 
    return rot_matrix

def calculate_magnet_pose(ur5_ee_pos,ur5_ee_rot):
    a = to_torch([[0,torch.pi/2,0],
              [0,torch.pi/2,0]])
    # ur5_ee_pos = ur5_ee_pos.squeeze()
    # ur5_ee_rot = ur5_ee_rot.squeeze()
    quat_rot = quat_from_euler_xyz(a[:,0],a[:,1],a[:,2])
    print(quat_rot.shape)
    magnet_rot_quat = quat_mul(quat_rot,ur5_ee_rot)
    magnet_rot_matrix = quat_to_rot_matrix(magnet_rot_quat)
    magnet_pos = ur5_ee_pos + magnet_rot_matrix[:,0,0:3]*0.1 # 64*3*3
    # magnet_pos = magnet_pos.unsqueeze(1)
    # magnet_rot_quat.unsqueeze(1)
    return magnet_pos,magnet_rot_quat

a = to_torch([[0.0,torch.pi,0],
              [torch.pi/10,torch.pi,0]])
# print(a[:,0].shape)
# b = quat_from_euler_xyz(a[:,0],a[:,1],a[:,2])
# c = to_torch([1,1,1]).unsqueeze(0)
# print(calculate_magnet_pose(c,b))

b = quat_from_euler_xyz(a[:,0],a[:,1],a[:,2])
print(b)

# c = torch.tensor([0,1,2])
# d = torch.tensor([4,5,6])
# print(d[c])


def force_moment(p, ma, mc):
    """
    计算永磁铁在空间中某位置对被驱动永磁铁产生的磁力和磁力矩
    :param pa: 永磁铁的位置 array 3*1
    :param pc: 空间中被驱动永磁铁的位置 array 3*1
    :param ma: 永磁铁的磁矩 array 3*1
    :param mc: 被驱动磁铁的磁矩 array 3*1
    :return: 磁力 force 3*1 和 磁力矩 moment 3*1
    """
    k = 4 * torch.pi * 1e-7
    ma = torch.tensor(ma).view(2,-1,1)
    mc = torch.tensor(mc).view(2,-1,1)
    p = torch.tensor(p).view(2,-1,1)
    p_norm = torch.norm(p,dim=1,keepdim=True)
    p_hat = p/p_norm
    p_trans = torch.transpose(p,1,2)
    p_hat_trans = torch.transpose(p_hat,1,2)
    ma_trans = torch.transpose(ma,1,2)
    eye_3 = torch.eye(3).unsqueeze(0).expand(2,3,3)

    # 磁源在某点产生的磁场强度
    field = k / (4*torch.pi*pow(p_norm,5)) * ((3*(p@p_trans)-pow(p_norm,2) * eye_3) @ ma)

    # print((torch.matmul((pow(p_norm,2)*eye_3),ma)).shape)

    # 磁源在某点产生的磁场梯度矩阵
    gradient = 3 * k / (4*torch.pi*pow(p_norm,4))*(ma@p_hat_trans+p_hat@ma_trans+(p_hat_trans@ma)*(eye_3-5*(p_hat@p_hat_trans)))
    
    force = torch.matmul(gradient,mc)

    S_mc = torch.tensor([[0, -mc[0,2, 0], mc[0,1, 0]],
                   [mc[0,2, 0], 0, -mc[0,0, 0]],
                   [-mc[0,1, 0], mc[0,0, 0], 0]])
    S_mc = torch.tensor(S_mc.unsqueeze(0).expand(2,3,3))
    # print(S_mc)

    # print(force)
    force = torch.matmul(gradient,mc)
    moment = torch.matmul(S_mc,field)
    print(force)
    print(moment)
    return force,moment


ma = torch.tensor([[0.0,0,26.2],
                    [0.0,0.0,26.2]])
mc = torch.tensor([[0.0,0.0,0.126],
                   [0.0,0.0,0.126]])
p = torch.tensor([[0.1,0.1,0.1],
                [0.2,0.2,0.2]])

force,torque = force_moment(p,ma,mc)

a = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
b = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
norm = torch.norm(a,p=2,dim=-1,keepdim=True)
a_hat = a/norm
print(torch.square(a)*a_hat)
print(norm**2*a_hat)
print(norm)

print(14 % 14)



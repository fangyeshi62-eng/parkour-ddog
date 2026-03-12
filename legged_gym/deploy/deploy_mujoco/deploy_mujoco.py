# # import time
# # import mujoco.viewer
# # import mujoco
# # import numpy as np
# # import torch
# # import yaml
# # import os

# # # 假设 LEGGED_GYM_ROOT_DIR 已定义，或者手动指定
# # LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # # --- 1. 高度采样逻辑 ---

# # def get_height_measurements(m, d, points_x, points_y):
# #     """
# #     获取相对于机器人的地面高度采样
# #     """
# #     base_pos = d.qpos[0:3]
# #     base_quat = d.qpos[3:7]
    
# #     # 建立旋转矩阵
# #     res_mat = np.zeros(9)
# #     mujoco.mju_quat2Mat(res_mat, base_quat)
# #     rot_mat = res_mat.reshape(3, 3)

# #     heights = np.zeros(len(points_x) * len(points_y), dtype=np.float32)
# #     ray_dir = np.array([0, 0, -1], dtype=np.float64) 
    
# #     count = 0
# #     for px in points_x:
# #         for py in points_y:
# #             # 局部坐标转世界坐标
# #             rel_sample_pos = np.array([px, py, 0.0])
# #             world_sample_pos = base_pos + rot_mat @ rel_sample_pos
# #             world_sample_pos[2] += 0.5 # 从上方发射射线
            
# #             # 射线检测
# #             dist = mujoco.mj_ray(m, d, world_sample_pos, ray_dir, None, 1, -1, np.zeros(1, dtype=np.int32))
            
# #             if dist > 0:
# #                 # 相对高度 = 地面绝对高度 - 机器人质心高度
# #                 h = (world_sample_pos[2] - dist) - base_pos[2]
# #                 # Go2站立高度约为0.28m，h在平地约为-0.28，加0.28使其归零
# #                 heights[count] = np.clip(h + 0.28, -1.2, 1.2)
# #             else:
# #                 heights[count] = -1.2
# #             count += 1
# #     return heights

# # # --- 2. 部署主程序 ---

# # if __name__ == "__main__":
# #     import argparse
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("config_file", type=str)
# #     args = parser.parse_args()
    
# #     # 加载配置
# #     current_dir = os.path.dirname(os.path.abspath(__file__))
# #     config_path = os.path.join(current_dir, "configs", args.config_file)
# #     with open(config_path, "r") as f:
# #         config = yaml.load(f, Loader=yaml.FullLoader)
        
# #     policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
# #     xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    
# #     # 初始化 MuJoCo
# #     m = mujoco.MjModel.from_xml_path(xml_path)
# #     d = mujoco.MjData(m)
# #     m.opt.timestep = config["simulation_dt"]
    
# #     # 加载模型
# #     device = "cpu"
# #     policy = torch.jit.load(policy_path).to(device)
# #     policy.eval()

# #     # 地形采样点
# #     measured_points_x = np.linspace(-0.5, 1.5, 21) 
# #     measured_points_y = np.linspace(-0.5, 0.5, 11) 
    
# #     # 控制参数
# #     kps = np.array(config["kps"], dtype=np.float32)
# #     kds = np.array(config["kds"], dtype=np.float32)
# #     default_angles = np.array(config["default_angles"], dtype=np.float32)
    
# #     obs = np.zeros(279, dtype=np.float32)
# #     action = np.zeros(12, dtype=np.float32)
# #     target_dof_pos = default_angles.copy()
# #     counter = 0
    
# #     d.qpos[2] = 0.4
# #     d.qpos[7:19] = default_angles  # 核心：让关节初始就在站立姿态
# #     mujoco.mj_forward(m, d)

# #     # 启动查看器
# #     with mujoco.viewer.launch_passive(m, d) as viewer:
# #         # 初始稳定：让狗子落到地上
# #         for _ in range(100):
# #             mujoco.mj_step(m, d)
# #             viewer.sync()

# #         while viewer.is_running():
# #             step_start = time.time()
            
# #             # 1. 执行 PD 控制 (每个物理步执行)
# #             tau = (target_dof_pos - d.qpos[7:]) * kps - d.qvel[6:] * kds
# #             d.ctrl[:] = tau
# #             mujoco.mj_step(m, d)

# #             # 2. 策略更新 (按控制频率执行)
# #             counter += 1
# #             if counter % config["control_decimation"] == 0:
# #                 quat = d.qpos[3:7].copy() # [w, x, y, z]
# #                 inv_quat = np.zeros(4)
# #                 mujoco.mju_negQuat(inv_quat, quat) # 获取四元数的逆

# #                 # --- 构造本体观测 (0:48) ---
                
# #                 # # 1. 局部线速度 (必须旋转)
# #                 # local_lin_vel = np.zeros(3)
# #                 # mujoco.mju_rotVecQuat(local_lin_vel, d.qvel[:3], inv_quat)
# #                 # obs[0:3] = local_lin_vel * config.get("lin_vel_scale", 2.0)
# #                 # 将世界系速度转为本体系局部速度
# #                 local_lin_vel = np.zeros(3)
# #                 mujoco.mju_rotVecQuat(local_lin_vel, d.sensor('base_lin_vel').data, inv_quat)
# #                 lin_vel_scale = config["lin_vel_scale"] 
# #                 obs[0:3] = local_lin_vel * lin_vel_scale
# #                 #obs[0:3] = local_lin_vel * config.get("lin_vel_scale", 2.0)
                
# #                 # 2. 局部角速度 (MuJoCo qvel[3:6] 已经是局部系)
# #                 # obs[3:6] = d.qvel[3:6] * config.get("ang_vel_scale", 0.25)
# #                 obs[3:6] = d.sensor('base_ang_vel').data * config.get("ang_vel_scale", 0.25)
                
# #                 # 3. 重力投影 (世界[0,0,-1]转局部)
# #                 world_gravity = np.array([0, 0, -1], dtype=np.float64)
# #                 local_gravity = np.zeros(3)
# #                 mujoco.mju_rotVecQuat(local_gravity, world_gravity, inv_quat)
# #                 obs[6:9] = local_gravity
                
# #                 # 4. 指令速度
# #                 obs[9:12] = np.array(config["cmd_init"]) * np.array(config["cmd_scale"])
# #                 # obs[9:12]=[0 , 0, 0]
# #                 # 5. 关节位置偏差
# #                 obs[12:24] = (d.qpos[7:] - default_angles) * config.get("dof_pos_scale", 1.0)
                
# #                 # 6. 关节速度
# #                 obs[24:36] = d.qvel[6:] * config.get("dof_vel_scale", 0.05)
                
# #                 # 7. 历史动作
# #                 obs[36:48] = action
                
# #                 # --- 构造地形观测 (48:279) ---
# #                 #obs[48:279] = get_height_measurements(m, d, measured_points_x, measured_points_y)
# #                 obs[48:279] = 0
# #                 # 诊断打印
# #                 if counter % 100 == 0:
# #                     print(f"\rV_lin: {local_lin_vel} | Grav: {local_gravity} | H_avg: {np.mean(obs[48:279]):.3f}", end="")
# #                     print(f"Joint Pos: {d.qpos[7:10]}") # 打印前三个关节位置
# #                     print(f"action: {action}")
# #                 # --- 模型推理 ---
# #                 obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(device)
# #                 with torch.no_grad():
# #                     # 注意：如果你的模型输出包含(action, next_state)，请使用 [0]
# #                     action = policy(obs_tensor).cpu().numpy().squeeze()
                
# #                 # 限制动作幅度，防止越界触发崩溃
# #                 action = np.clip(action, -5.0, 5.0)
                
# #                 # 计算机标关节位置
# #                 target_dof_pos = action * config.get("action_scale", 0.25) + default_angles
# #             # if viewer.is_running():
# #             #     with viewer.lock():
# #             #         viewer.user_scn.ngeom = 0
# #             #         base_pos = d.qpos[0:3]
# #             #         base_quat = d.qpos[3:7]
# #             #         res_mat = np.zeros(9)
# #             #         mujoco.mju_quat2Mat(res_mat, base_quat)
# #             #         rot_mat = res_mat.reshape(3, 3)
                    
# #             #         c = 0
# #             #         for px in measured_points_x:
# #             #             for py in measured_points_y:
# #             #                 # 获取高度图中记录的相对高度并还原
# #             #                 h_val = obs[48 + c] # 从 obs 数组里直接取
# #             #                 rel_h = h_val - 0.28
                            
# #             #                 world_pos = base_pos + rot_mat @ np.array([px, py, rel_h])
                            
# #             #                 # 在地面位置画一个小红点
# #             #                 mujoco.mjv_initGeom(
# #             #                     viewer.user_scn.geoms[viewer.user_scn.ngeom],
# #             #                     type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.015, 0, 0], 
# #             #                     pos=world_pos, mat=np.eye(3).flatten(), rgba=[1, 0, 0, 1]
# #             #                 )
# #             #                 viewer.user_scn.ngeom += 1
# #             #                 c += 1
# #             viewer.sync()
        
# #             # 帧率同步
# #             time_until_next_step = m.opt.timestep - (time.time() - step_start)
# #             if time_until_next_step > 0:
# #                 time.sleep(time_until_next_step)

# import time
# import mujoco.viewer
# import mujoco
# import numpy as np
# import torch
# import yaml
# import os

# def get_gravity_orientation(rot_mat_transpose):
#     """ 获取机体系下的重力投影 [0, 0, -1] """
#     # 旋转矩阵 R 的转置乘 [0, 0, -1]，等同于取 R^T 的第三列并取负
#     return rot_mat_transpose @ np.array([0.0, 0.0, -1.0])

# def pd_control(target_q, q, kp, target_dq, dq, kd):
#     return (target_q - q) * kp + (target_dq - dq) * kd

# def get_height_measurements(m, d, rot_mat, points_x, points_y):
#     """ 
#     地形采样逻辑：匹配 LeggedRobot._get_heights()
#     计算 (base_z + offset) - ground_z
#     """
#     base_pos = d.qpos[0:3]
#     heights = np.zeros(len(points_x) * len(points_y), dtype=np.float32)
#     ray_dir = np.array([0, 0, -1], dtype=np.float64) 
    
#     # 按照 Go2RoughCfg 默认，通常不加额外 offset 或根据训练设定
#     # 这里直接计算相对高度差
#     count = 0
#     for px in points_x:
#         for py in points_y:
#             rel_sample_pos = np.array([px, py, 0.0])
#             # 将采样点位置变换到世界坐标系
#             world_sample_pos = base_pos + rot_mat @ rel_sample_pos
#             world_sample_pos[2] += 1.0 # 从机身上方1米处向下发射射线
            
#             dist = mujoco.mj_ray(m, d, world_sample_pos, ray_dir, None, 1, -1, np.zeros(1, dtype=np.int32))
            
#             if dist > 0:
#                 ground_z = world_sample_pos[2] - dist
#                 #h = base_pos[2] - ground_z
#                 h = (ground_z - base_pos[2]) + 0.28
#                 heights[count] = np.clip(h, -1.0, 1.0)
#             else:
#                 heights[count] = -1.0
#             count += 1
#     return heights

# if __name__ == "__main__":
#     # --- 1. 参数与路径配置 ---
#     # 建议将这些放入你的 g2.yaml
#     config = {
#         "policy_path": "/root/parkour/legged_gym/deploy/pre_train/g2/policy_gru_22.pt",
#         "xml_path": "/root/parkour/legged_gym/resources/robots/go2/scene.xml",
#         "simulation_dt": 0.002,
#         "control_decimation": 4,
#         "kps": [40.0] * 12,
#         "kds": [1.0] * 12,
#         "action_scale": 0.5,
#         "lin_vel_scale": 2.0,
#         "ang_vel_scale": 0.25,
#         "dof_pos_scale": 1.0,
#         "dof_vel_scale": 0.05,
#         "default_angles": [0.1, 0.7, -1.5, -0.1, 0.7, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]
#     }

#     # --- 2. 环境初始化 ---
#     m = mujoco.MjModel.from_xml_path(config["xml_path"])
#     d = mujoco.MjData(m)
#     m.opt.timestep = config["simulation_dt"]
    
#     # 初始化机器人状态
#     d.qpos[2] = 0.4  # 起始高度
#     d.qpos[7:19] = config["default_angles"]
#     mujoco.mj_forward(m, d)

#     # 加载策略 (带 GRU 状态初始化)
#     device = "cpu"
#     policy = torch.jit.load(config["policy_path"]).to(device)
#     hidden_state = None  # 如果是 GRU 模型，需要维护隐藏状态

#     # 地形采样点
#     measured_points_x = np.linspace(-0.5, 1.5, 21) 
#     measured_points_y = np.linspace(-0.5, 0.5, 11) 

#     # 运行变量
#     obs = np.zeros(279, dtype=np.float32)
#     action = np.zeros(12, dtype=np.float32)
#     target_dof_pos = np.array(config["default_angles"])
#     counter = 0

#     with mujoco.viewer.launch_passive(m, d) as viewer:
#         while viewer.is_running():
#             step_start = time.time()

#             # --- 3. 物理仿真与控制 ---
#             # 软启动：KP 在前 1 秒内从 0 升到 40，防止开场爆炸
#             kp_ramp = min(1.0, d.time / 1.0)
#             current_kps = np.array(config["kps"]) * kp_ramp
            
#             tau = pd_control(target_dof_pos, d.qpos[7:19], current_kps, 0, d.qvel[6:18], config["kds"])
#             d.ctrl[:] = tau
#             mujoco.mj_step(m, d)

#             # --- 4. 策略推断 (50Hz) ---
#             if counter % config["control_decimation"] == 0:
#                 # 获取旋转矩阵 R
#                 res_mat = np.zeros(9)
#                 mujoco.mju_quat2Mat(res_mat, d.qpos[3:7])
#                 rot_mat = res_mat.reshape(3, 3)
#                 rot_mat_T = rot_mat.T # 机体坐标系基向量

#                 # 构造观测 (严格遵循训练顺序)
#                 # 1. 机体系线速度 (World -> Local)
#                 obs[0:3] = (rot_mat_T @ d.qvel[:3]) * config["lin_vel_scale"]
                
#                 # 2. 机体系角速度
#                 obs[3:6] = d.qvel[3:6] * config["ang_vel_scale"]
                
#                 # 3. 重力投影
#                 obs[6:9] = get_gravity_orientation(rot_mat_T)
                
#                 # 4. 指令 (示例：前进 0.5m/s)
#                 obs[9:12] = np.array([0.5, 0.0, 0.0]) * np.array([2.0, 2.0, 0.25])
                
#                 # 5. 关节位置偏差
#                 obs[12:24] = (d.qpos[7:19] - config["default_angles"]) * config["dof_pos_scale"]
                
#                 # 6. 关节速度
#                 obs[24:36] = d.qvel[6:18] * config["dof_vel_scale"]
                
#                 # 7. 历史动作
#                 obs[36:48] = action
                
#                 # 8. 地形采样
#                 #obs[48:279] = get_height_measurements(m, d, rot_mat, measured_points_x, measured_points_y)
#                 obs[48:279] = 0
#                 # 模型推理
#                 obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
#                 with torch.no_grad():
#                     # 处理 GRU 模型：通常输入是 (obs, hidden_state)
#                     # 如果你的模型导出时包含 hidden_state，请按需修改
#                     res = policy(obs_tensor)
#                     if isinstance(res, tuple):
#                         action = res[0].cpu().numpy().squeeze()
#                     else:
#                         action = res.cpu().numpy().squeeze()
                
#                 # 映射到目标角度
#                 target_dof_pos = action * config["action_scale"] + config["default_angles"]
#                 if counter % 100 == 0:
#                     print("-" * 50)
#                     print(f"Time: {d.time:.2f}s | Step Counter: {counter}")
                    
#                     # 1. 运动学状态
#                     lin_vel = rot_mat_T @ d.qvel[:3]
#                     print(f"Base Lin Vel (Local): x={lin_vel[0]:.3f}, y={lin_vel[1]:.3f}, z={lin_vel[2]:.3f}")
                    
#                     # 2. 控制参数
#                     print(f"Current KP Ramp: {kp_ramp:.2f}")
                    
#                     # 3. 观测与动作预览
#                     # 取前12个obs(速度、角速度、重力、指令)
#                     print(f"Obs (First 12): {np.round(obs[:12], 3)}")
#                     # 动作输出 (Action 通常在 -1 到 1 之间)
#                     print(f"Action (First 3): {np.round(action[:3], 3)}")
                    
#                     # 4. 关节位置偏差 (判断机器人是否在跟踪目标)
#                     # pos_err = np.mean(np.abs(target_dof_pos - current_qpos))
#                     # print(f"Avg Joint Pos Error: {pos_err:.4f} rad")
#             counter += 1
#             viewer.sync()

#             # 时间同步
#             time_until_next = config["simulation_dt"] - (time.time() - step_start)
#             if time_until_next > 0:
#                 time.sleep(time_until_next)
import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import os

# # # 假设 LEGGED_GYM_ROOT_DIR 已定义，或者手动指定
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_gravity_orientation(quaternion):
    # 保持你原始的四元数投影逻辑
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    # --- 1. 参数配置 ---
    num_actions, num_obs = 12, 279
    lin_vel_scale, ang_vel_scale = 2.0, 0.25
    dof_pos_scale, dof_vel_scale = 1.0, 0.05
    action_scale = 0.5 
    
    default_angles = np.array([0.1, 0.7, -1.5, -0.1, 0.7, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.float32)
    
    # 初始参数优化
    target_kps = np.full(12, 40.0) 
    kds = np.full(12, 1)          # 稍微调高阻尼，有助于吸收冲击
    control_decimation = 10
    simulation_dt = 0.002 

    # --- 2. 环境初始化 ---
    model_path = "/root/parkour/legged_gym/resources/robots/go2/scene.xml"
    policy_path = "/root/parkour/legged_gym/deploy/pre_train/g2/policy_gru_22.pt"
    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    # 【补丁 1：精准初始化，防止穿模排斥】
    d.qpos[7:] = default_angles
    d.qpos[2] = 0.445              # Go2 身体中心离地约 0.3-0.34m
    d.qvel[:] = 0                 # 初始速度彻底清零
    mujoco.mj_forward(m, d)       # 计算运动学，消除初始应力

    policy = torch.jit.load(policy_path)
    obs = np.zeros(num_obs, dtype=np.float32)
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    
    # 指令设定
    target_cmd = np.array([0.25 , 0, 0]) 
    
    counter = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # 【补丁 2：KP 爬坡逻辑，防止开场“炸飞”】
            # 在前 1.0 秒内，KP 从 0 逐渐增加到 20
            # 这让机器人像慢慢醒过来，而不是突然被电击
            current_kp_scale = min(1.0, d.time / 1.0)
            current_kps = target_kps * current_kp_scale
            
            # --- 3. 物理仿真 step ---
            tau = pd_control(target_dof_pos, d.qpos[7:], current_kps, 0, d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            
            # --- 4. 策略推断 ---
            if counter % control_decimation == 0:
                res_mat = np.zeros(9)
                mujoco.mju_quat2Mat(res_mat, d.qpos[3:7])
                rot_mat = res_mat.reshape(3, 3)
                rot_mat_T = rot_mat.T # 机体坐标系基向量

                qj, dqj = d.qpos[7:], d.qvel[6:]
                quat, omega = d.qpos[3:7], d.qvel[3:6]
                lin_vel = d.qvel[:3]
                # 获取传感器线速度
                #base_lin_vel = d.sensor('base_lin_vel').data.copy()
                
                # 【补丁 3：初始观测过滤】
                # 前 0.5 秒即使身体有晃动，我们也告诉网络速度为 0，防止它产生过大的纠偏动作
                if d.time < 0.5:
                    input_lin_vel_world = np.zeros(3)
                    cmd = np.zeros(3)
                else:
                    input_lin_vel_world = d.qvel[:3]
                    cmd = target_cmd

                # 构造 Observation
                obs[0:3] =  (rot_mat_T @ input_lin_vel_world)* lin_vel_scale 
                obs[3:6] = omega * ang_vel_scale
                obs[6:9] = get_gravity_orientation(quat)
                obs[9:12] = cmd * np.array([2, 2, 0.25])
                obs[12:24] = (qj - default_angles) * dof_pos_scale
                obs[24:36] = dqj * dof_vel_scale
                obs[36:48] = action
                obs[48:279]= 0
                
                # 推理
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                with torch.no_grad():
                    new_action = policy(obs_tensor).detach().numpy().squeeze()
                action = new_action.copy()
                # 只有在 KP 稳定后才允许动作大幅度更新
                if d.time > 0.2:
                    target_dof_pos = action * action_scale + default_angles

                # 打印调试信息
                if counter % 100 == 0:
                     print(f"Time: {d.time:.2f} | KP_Scale: {current_kp_scale:.2f} | Z-Vel: {lin_vel[2]:.2f}")
                     print(obs[0:3])
            counter += 1
            viewer.sync()
            
            # 频率控制
            time_to_sleep = simulation_dt - (time.time() - step_start)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
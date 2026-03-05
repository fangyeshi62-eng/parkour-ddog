import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import os

# 假设 LEGGED_GYM_ROOT_DIR 已定义，或者手动指定
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 1. 高度采样逻辑 ---

def get_height_measurements(m, d, points_x, points_y):
    """
    获取相对于机器人的地面高度采样
    """
    base_pos = d.qpos[0:3]
    base_quat = d.qpos[3:7]
    
    # 建立旋转矩阵
    res_mat = np.zeros(9)
    mujoco.mju_quat2Mat(res_mat, base_quat)
    rot_mat = res_mat.reshape(3, 3)

    heights = np.zeros(len(points_x) * len(points_y), dtype=np.float32)
    ray_dir = np.array([0, 0, -1], dtype=np.float64) 
    
    count = 0
    for px in points_x:
        for py in points_y:
            # 局部坐标转世界坐标
            rel_sample_pos = np.array([px, py, 0.0])
            world_sample_pos = base_pos + rot_mat @ rel_sample_pos
            world_sample_pos[2] += 0.5 # 从上方发射射线
            
            # 射线检测
            dist = mujoco.mj_ray(m, d, world_sample_pos, ray_dir, None, 1, -1, np.zeros(1, dtype=np.int32))
            
            if dist > 0:
                # 相对高度 = 地面绝对高度 - 机器人质心高度
                h = (world_sample_pos[2] - dist) - base_pos[2]
                # Go2站立高度约为0.28m，h在平地约为-0.28，加0.28使其归零
                heights[count] = np.clip(h + 0.28, -1.2, 1.2)
            else:
                heights[count] = -1.2
            count += 1
    return heights

# --- 2. 部署主程序 ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()
    
    # 加载配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "configs", args.config_file)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    
    # 初始化 MuJoCo
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]
    
    # 加载模型
    device = "cpu"
    policy = torch.jit.load(policy_path).to(device)
    policy.eval()

    # 地形采样点
    measured_points_x = np.linspace(-0.5, 1.5, 21) 
    measured_points_y = np.linspace(-0.5, 0.5, 11) 
    
    # 控制参数
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    
    obs = np.zeros(279, dtype=np.float32)
    action = np.zeros(12, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    counter = 0
    
    d.qpos[2] = 0.4
    d.qpos[7:19] = default_angles  # 核心：让关节初始就在站立姿态
    mujoco.mj_forward(m, d)

    # 启动查看器
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 初始稳定：让狗子落到地上
        for _ in range(100):
            mujoco.mj_step(m, d)
            viewer.sync()

        while viewer.is_running():
            step_start = time.time()
            
            # 1. 执行 PD 控制 (每个物理步执行)
            tau = (target_dof_pos - d.qpos[7:]) * kps - d.qvel[6:] * kds
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            # 2. 策略更新 (按控制频率执行)
            counter += 1
            if counter % config["control_decimation"] == 0:
                quat = d.qpos[3:7].copy() # [w, x, y, z]
                inv_quat = np.zeros(4)
                mujoco.mju_negQuat(inv_quat, quat) # 获取四元数的逆

                # --- 构造本体观测 (0:48) ---
                
                # 1. 局部线速度 (必须旋转)
                local_lin_vel = np.zeros(3)
                mujoco.mju_rotVecQuat(local_lin_vel, d.qvel[:3], inv_quat)
                obs[0:3] = local_lin_vel * config.get("lin_vel_scale", 2.0)
                
                # 2. 局部角速度 (MuJoCo qvel[3:6] 已经是局部系)
                obs[3:6] = d.qvel[3:6] * config.get("ang_vel_scale", 0.25)
                
                # 3. 重力投影 (世界[0,0,-1]转局部)
                world_gravity = np.array([0, 0, -1], dtype=np.float64)
                local_gravity = np.zeros(3)
                mujoco.mju_rotVecQuat(local_gravity, world_gravity, inv_quat)
                obs[6:9] = local_gravity
                
                # 4. 指令速度
                # obs[9:12] = np.array(config["cmd_init"]) * np.array(config["cmd_scale"])
                obs[9:12]=[1.5 , -1, 0]
                # 5. 关节位置偏差
                obs[12:24] = (d.qpos[7:] - default_angles) * config.get("dof_pos_scale", 1.0)
                
                # 6. 关节速度
                obs[24:36] = d.qvel[6:] * config.get("dof_vel_scale", 0.05)
                
                # 7. 历史动作
                obs[36:48] = action
                
                # --- 构造地形观测 (48:279) ---
                obs[48:279] = get_height_measurements(m, d, measured_points_x, measured_points_y)

                # 诊断打印
                if counter % 100 == 0:
                    print(f"\rV_lin: {local_lin_vel} | Grav: {local_gravity} | H_avg: {np.mean(obs[48:279]):.3f}", end="")
                    print(f"Joint Pos: {d.qpos[7:10]}") # 打印前三个关节位置
                    print(f"action: {action}")
                # --- 模型推理 ---
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    # 注意：如果你的模型输出包含(action, next_state)，请使用 [0]
                    action = policy(obs_tensor).cpu().numpy().squeeze()
                
                # 限制动作幅度，防止越界触发崩溃
                action = np.clip(action, -5.0, 5.0)
                
                # 计算机标关节位置
                target_dof_pos = action * config.get("action_scale", 0.25) + default_angles
            # if viewer.is_running():
            #     with viewer.lock():
            #         viewer.user_scn.ngeom = 0
            #         base_pos = d.qpos[0:3]
            #         base_quat = d.qpos[3:7]
            #         res_mat = np.zeros(9)
            #         mujoco.mju_quat2Mat(res_mat, base_quat)
            #         rot_mat = res_mat.reshape(3, 3)
                    
            #         c = 0
            #         for px in measured_points_x:
            #             for py in measured_points_y:
            #                 # 获取高度图中记录的相对高度并还原
            #                 h_val = obs[48 + c] # 从 obs 数组里直接取
            #                 rel_h = h_val - 0.28
                            
            #                 world_pos = base_pos + rot_mat @ np.array([px, py, rel_h])
                            
            #                 # 在地面位置画一个小红点
            #                 mujoco.mjv_initGeom(
            #                     viewer.user_scn.geoms[viewer.user_scn.ngeom],
            #                     type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.015, 0, 0], 
            #                     pos=world_pos, mat=np.eye(3).flatten(), rgba=[1, 0, 0, 1]
            #                 )
            #                 viewer.user_scn.ngeom += 1
            #                 c += 1
            viewer.sync()
        
            # 帧率同步
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)



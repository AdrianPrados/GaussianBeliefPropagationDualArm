from Adam_sim.scripts.adam import ADAM
import time
import pybullet as p
import pybullet_data
import scipy.io
#from Adam_sim.scripts.node_connection import Node
#import rospy

import numpy as np
import itertools

from motionGBP.obstacle3D import ObstacleMap3D
from motionGBP.agent3D import Agent3D, Env

import math
import pandas as pd
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


dt = 1./240.

def plot_paths_3d(finalPathR, path_agent1_EE):
    finalPathR = np.array(finalPathR)
    path_agent1_EE = np.array(path_agent1_EE)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(finalPathR[:, 0]*100, finalPathR[:, 1]*100, finalPathR[:, 2]*100, label='finalPathR', color='red')
    ax.plot(path_agent1_EE[:, 0], path_agent1_EE[:, 1], path_agent1_EE[:, 2], label='path_agent1_EE', color='blue')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.show()
def plot_final_velocity(finalVelocity):

    finalVelocity = np.array(finalVelocity)  
    time = np.arange(len(finalVelocity))  

    plt.figure(figsize=(8, 5))
    
    for i in range(finalVelocity.shape[1]):
        plt.plot(time, finalVelocity[:, i], label=f'Velocidad axes {i+1}')

    plt.xlabel('Time (Index)')
    plt.ylabel('Velocity')
    plt.title('End effector Velocity')
    plt.legend()
    plt.grid(True)
    plt.show()

def load_data(file_name, step=1):
    data = pd.read_csv(file_name)
    pose = []
    orientation = []
    tool0_pose = []

    tool_offset = np.array([0, 0, 8])  

    for i in range(0, len(data), step):  
        row = data.iloc[i]  
        position = np.array([row['x'] * 100, row['y'] * 100, row['z'] * 100]) 
        ori = np.array([row['qx'], row['qy'], row['qz'], row['qw']]) 
        
        rotation_matrix = R.from_quat(ori).as_matrix()

        tool_offset_rotated = rotation_matrix @ tool_offset  
        
        tool0_position = position + tool_offset_rotated  

        pose.append(position)
        orientation.append(ori)
        tool0_pose.append(tool0_position)

    return pose, orientation, tool0_pose


#* --- PyBullet Configuaration ---


if __name__ == '__main__':
    robot_urdf_path = "/home/adrian/Escritorio/ImitationLearning/GaussianBelief/src/Adam_sim/paquetes_simulacion/rb1_base_description/robots/robotDummy.urdf"
    robot_stl_path = "/home/adrian/Escritorio/ImitationLearning/GaussianBelief/src/Adam_sim/paquetes_simulacion/rb1_base_description/meshes/others/adam_model.stl"
    
    adam = ADAM(robot_urdf_path,robot_stl_path,0,1)
    
    
    
    p.setRealTimeSimulation(adam.useRealTimeSimulation)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    """ initial_left_pose = [0.11,-1.96,-0.79,0.67,-0.08,-0.01]
    initial_right_pose = [0.66,-2.26,-0.70,-0.14,2.55,-1.15] """
    #initial_left_pose = [-0.85,-1.72,0.042,-1.036,1.0249,1.911]
    #initial_right_pose = [-0.43,-1.83,0.681,-1.43,1.908,-1.53]
    initial_right_pose = [2.7354,-1.5973,-1.1237,-1.2453,0.4834,-0.1599]
    initial_left_pose = [-2.0935,-2.0019,-0.8266,-2.6136,4.4560,5.1101]
    
    
    
    # Initial pose
    for _ in range(100):
        adam.kinematics.initial_arm_pose("right",initial_right_pose)
        adam.kinematics.initial_arm_pose("left",initial_left_pose)
    
    
    #! Init of GBP

    # --- Obstacles: red sphere ---
    obstacle_radius = 0.1
    obstacle_position = [0.45, 0.2, 0.8]  # [x, y, z] in cm
    omap = ObstacleMap3D()
    omap2 = ObstacleMap3D()
    #omap.set_sphere('', 45, 20, 80, 8)
    #omap2.set_sphere('', 45, 20, 80, 8) 
    

    obstacleCollisionShape = p.createCollisionShape(p.GEOM_SPHERE, radius=obstacle_radius)
    obstacleVisualShape = p.createVisualShape(p.GEOM_SPHERE, radius=obstacle_radius, rgbaColor=[1, 0, 0, 0.5])
    
    """ obstacleId = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=obstacleCollisionShape,
                                baseVisualShapeIndex=obstacleVisualShape,
                                basePosition=obstacle_position) """

    #* --- Load trajectories generated with LfD ---
    file_name2 = "/home/adrian/Escritorio/ImitationLearning/GaussianBelief/src/data/Cubos1Der.csv"
    file_name1 = "/home/adrian/Escritorio/ImitationLearning/GaussianBelief/src/data/Cubos1Izq.csv"
    path_agent0_EE, orn0, path_agent0 = load_data(file_name1)
    path_agent1_EE, orn1,path_agent1 = load_data(file_name2)
    
    #! Use this if you wanna use the EE of the arm, if you use the EE of the hand, use path_agent0 directly
    path_agent0=path_agent0_EE
    path_agent1=path_agent1_EE
    
    #* --- Create agents ---
    agent_radius = 0.08
    
    initial_position0 = [path_agent0[0][0], path_agent0[0][1], path_agent0[0][2], 0, 0, 0]  
    final_position0   = [path_agent0[-1][0], path_agent0[-1][1], path_agent0[-1][2], 10, 20, 5]
    
    initial_position1 = [path_agent1[0][0], path_agent1[0][1], path_agent1[0][2], 0, 0, 0]  
    final_position1   = [path_agent1[-1][0], path_agent1[-1][1], path_agent1[-1][2], 10, 20, 5]


    # Agent spheres
    agentCollisionShape = p.createCollisionShape(p.GEOM_SPHERE, radius=agent_radius)
    agentVisualShape = p.createVisualShape(p.GEOM_SPHERE, radius=agent_radius, rgbaColor=[0, 1, 0, 0.6])
    
    agentCollisionShape2 = p.createCollisionShape(p.GEOM_SPHERE, radius=agent_radius)
    agentVisualShape2 = p.createVisualShape(p.GEOM_SPHERE, radius=agent_radius, rgbaColor=[0, 0, 1, 0.6])

    agentId = p.createMultiBody(baseMass=0,
                                baseVisualShapeIndex=agentVisualShape,
                                baseCollisionShapeIndex=-1,
                                basePosition=path_agent0[0])
    
    agentId2 = p.createMultiBody(baseMass=0,
                                baseVisualShapeIndex=agentVisualShape2,
                                baseCollisionShapeIndex=-1,
                                basePosition=path_agent1[0])
    
    agent0 = Agent3D('a0', initial_position0, final_position0, steps=12, radius=8, omap=omap2, path=path_agent0)
    agent1 = Agent3D('a1', initial_position1, final_position1, steps=12, radius=8, omap=omap, path=path_agent1)
    env = Env()
    env.add_agent(agent0)
    env.add_agent(agent1)

    # --- Draw trajectories ---
    for i in range(len(path_agent0) - 1):
        start = (path_agent0[i][0]/100, path_agent0[i][1]/100, path_agent0[i][2]/100)
        end = (path_agent0[i+1][0]/100, path_agent0[i+1][1]/100, path_agent0[i+1][2]/100)
        p.addUserDebugLine(start, end, lineColorRGB=[0, 1, 0], lineWidth=2, lifeTime=0)

    for i in range(len(path_agent1) - 1):
        start = (path_agent1[i][0]/100, path_agent1[i][1]/100, path_agent1[i][2]/100)
        end = (path_agent1[i+1][0]/100, path_agent1[i+1][1]/100, path_agent1[i+1][2]/100)
        p.addUserDebugLine(start, end, lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=0)



    #? === Dynamic obstacles ===

    arm1_obstacle_ids = []
    arm1_obstacle_names = []
    finalPathR = []
    finalVelocity = []
    sphere_radius = 0.09  
    for i in adam.ur3_left_arm_joints:
        linkState = p.getLinkState(adam.robot_id, i)
        pos_link = linkState[4]
        name = "arm1_link_" + str(i)
        omap.set_sphere(name, pos_link[0]*100, pos_link[1]*100, pos_link[2]*100, sphere_radius*100)
        obs_visual = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=[1, 1, 0, 0.8])
        obs_id = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=-1,  # solo visual
                                baseVisualShapeIndex=obs_visual,
                                basePosition=pos_link)
        arm1_obstacle_ids.append(obs_id)
        arm1_obstacle_names.append(name)
        
    arm2_obstacle_ids = []
    arm2_obstacle_names = []
    for i in adam.ur3_right_arm_joints:
        linkState = p.getLinkState(adam.robot_id, i)
        pos_link = linkState[4]
        name = "arm2_link_" + str(i)
        omap2.set_sphere(name, pos_link[0]*100, pos_link[1]*100, pos_link[2]*100, sphere_radius*100)
        obs_visual = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=[1, 1, 0, 0.8])
        obs_id = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=-1,
                                baseVisualShapeIndex=obs_visual,
                                basePosition=pos_link)
        arm2_obstacle_ids.append(obs_id)
        arm2_obstacle_names.append(name)
    
    
    prev_orn0 = orn0[0]
    prev_orn1 = orn1[0]
    
    adam.print_robot_info()
    #time.sleep(1000)
    adam.handkinematics.hand_forward_kinematics("both", [900,0,0,0,0,0], [900,0,0,0,0,0])
    
    table_id = p.loadURDF("table/table.urdf",basePosition=[0.75, 0, 0.03], baseOrientation= p.getQuaternionFromEuler([0, 0, 1.57]),useFixedBase=True, globalScaling=1.17)
    
    #! Cilindro
    """ collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.04, height=0.26)
    visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.04, length=0.26, rgbaColor=[0.5, 0.5, 0.5, 1]) """
    
    #! Cubo
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[1, 1, 0, 1])
    
    aabb_min, aabb_max = p.getAABB(table_id)
    
    #* Cubes right hand
    cube_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=collision_shape, 
        baseVisualShapeIndex=visual_shape, 
        basePosition=[0.57, -0.52, aabb_max[2]],
        baseOrientation= p.getQuaternionFromEuler([0, 0, 0.7])
    )
    cube_id2 = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=collision_shape, 
        baseVisualShapeIndex=visual_shape, 
        basePosition=[0.55, -0.4, aabb_max[2]],
        baseOrientation= p.getQuaternionFromEuler([0, 0, 0.7])
    )
    
    #* Cubes left hand
    cube_id3 = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=collision_shape, 
        baseVisualShapeIndex=visual_shape, 
        basePosition=[0.54, 0.1, aabb_max[2]],
        baseOrientation= p.getQuaternionFromEuler([0, 0, 0])
    )
    cube_id4 = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=collision_shape, 
        baseVisualShapeIndex=visual_shape, 
        basePosition=[0.63, 0.22, aabb_max[2]],
        baseOrientation= p.getQuaternionFromEuler([0, 0, 0.7])
    )
    
    #* Cubo Ref
    cube_id_fixed = p.createMultiBody(
        baseMass=10000000000, 
        baseCollisionShapeIndex=collision_shape, 
        baseVisualShapeIndex=visual_shape, 
        basePosition=[0.40, -0.12, aabb_max[2]]  # Un poco sobre el suelo
    )

    width  = aabb_min[0]
    depth  = aabb_max[1] - aabb_min[1]
    height = aabb_max[2]
    
    derState = 0
    izqState = 0
    
    IkR = []
    IkL = []
    while True:
        #* We simulate the grasping of the cubes
        keys = p.getKeyboardEvents()
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            if cube_id is not None:
                adam.handkinematics.hand_forward_kinematics("right", [900,0,900,900,900,900])
                if derState == 0:
                    p.removeBody(cube_id)
                elif derState == 1:
                    p.removeBody(cube_id2)
                cube_id = None
            else:
                if derState == 0:
                    adam.handkinematics.hand_forward_kinematics("right", [900,0,1,1,1,1])
                    time.sleep(0.5)
                    cube_id = p.createMultiBody(
                    baseMass=0.01, 
                    baseCollisionShapeIndex=collision_shape, 
                    baseVisualShapeIndex=visual_shape, 
                    basePosition=[0.40, -0.07, aabb_max[2]] )
                    derState = 1
                elif derState == 1:
                    adam.handkinematics.hand_forward_kinematics("right", [900,0,1,1,1,1])
                    time.sleep(0.5)
                    cube_id2 = p.createMultiBody(
                    baseMass=0.01, 
                    baseCollisionShapeIndex=collision_shape, 
                    baseVisualShapeIndex=visual_shape, 
                    basePosition=[0.40, -0.095, aabb_max[2]+0.05] )
                    derState = 2
        
        if ord('l') in keys and keys[ord('l')] & p.KEY_WAS_TRIGGERED:
            if cube_id4 is not None:
                adam.handkinematics.hand_forward_kinematics("left", [900,0,900,900,900,900])
                if izqState == 0:
                    p.removeBody(cube_id4)
                elif izqState == 1:
                    p.removeBody(cube_id3)
                cube_id4 = None
            else:
                if izqState == 0:
                    adam.handkinematics.hand_forward_kinematics("left", [900,0,1,1,1,1])
                    time.sleep(0.5)
                    cube_id4 = p.createMultiBody(
                    baseMass=0.01, 
                    baseCollisionShapeIndex=collision_shape, 
                    baseVisualShapeIndex=visual_shape, 
                    basePosition=[0.40, -0.14, aabb_max[2]] )
                    izqState = 1
                elif izqState == 1:
                    adam.handkinematics.hand_forward_kinematics("left", [900,0,1,1,1,1])
                    time.sleep(0.5)
                    cube_id3 = p.createMultiBody(
                    baseMass=0.01, 
                    baseCollisionShapeIndex=collision_shape, 
                    baseVisualShapeIndex=visual_shape, 
                    basePosition=[0.40, -0.15, aabb_max[2]+0.05] )
                    izqState = 2
        
        #!--- Implementation logic----
        
        try:
            for i in adam.ur3_left_arm_joints:
                linkState = p.getLinkState(adam.robot_id, i)
                pos_link = linkState[4]
                p.resetBasePositionAndOrientation(arm1_obstacle_ids[i-45], pos_link, [0, 0, 0, 1])
                # Actualizar en omap (asumimos que omap.objects es un diccionario modificable)
                omap.objects[arm1_obstacle_names[i-45]]['centerx'] = pos_link[0]*100
                omap.objects[arm1_obstacle_names[i-45]]['centery'] = pos_link[1]*100
                omap.objects[arm1_obstacle_names[i-45]]['centerz'] = pos_link[2]*100
                
            for i in adam.ur3_right_arm_joints:
                linkState = p.getLinkState(adam.robot_id, i)
                pos_link = linkState[4]
                p.resetBasePositionAndOrientation(arm2_obstacle_ids[i-20], pos_link, [0, 0, 0, 1])
                omap2.objects[arm2_obstacle_names[i-20]]['centerx'] = pos_link[0]*100
                omap2.objects[arm2_obstacle_names[i-20]]['centery'] = pos_link[1]*100
                omap2.objects[arm2_obstacle_names[i-20]]['centerz'] = pos_link[2]*100
            
            env.step_plan()
            
            for agent in env._agents:
                cont = False
                state = agent.get_state()
                if state[0] is None:
                    continue
                
                x, y, z, vx, vy, vz = state[0]
                pos = [x/100,y/100,z/100]
                vel = [vx/100,vy/100,vz/100]

                if agent._name == 'a0':
                    if agent._change == 1:
                        try:
                            orn = orn0[agent._current_index]
                        except:
                            orn = orn0[-1]
                        print("Orientation in the path",orn)
                    else :
                        print("Previous orientation: ",prev_orn0)
                        orn = prev_orn0
                    
                    poses = [tuple(pos), orn]

                    while cont == False:
                        ikLeft,_,_ = adam.kinematics.move_arm_to_pose("left", poses,dummy=False)
                        IkL.append(ikLeft[20:26])
                        """ link_state = p.getLinkState(adam.robot_id, 54)
                        currentEEPos,ori = link_state[4],link_state[5] """
                        currentEEPos,ori = adam.kinematics.calculate_arm_forward_kinematics("left")
                        
                        """ print("Posición del efector final:", currentEEPos)
                        print("Posicion deseada del efector final:", pos) """
                        diff =np.linalg.norm(np.array(currentEEPos) - np.array(pos))
                        if  diff < 0.05:
                            cont = True
                            pos_I = pos
                            stateI=state
                        else:
                            cont = False
                    print("Left:",agent._current_index)
                    p.resetBasePositionAndOrientation(agentId, pos_I, [0, 0, 0, 1])
                    p.resetBaseVelocity(agentId, linearVelocity=vel)
                    
                    for s0, s1 in itertools.pairwise(stateI):
                        if s0 is None or s1 is None:
                            break
                        x0, y0, z0, *_ = s0
                        x1, y1, z1, *_ = s1
                        p.addUserDebugLine([x0/100, y0/100, z0/100], [x1/100, y1/100, z1/100],
                                            lineColorRGB=[0, 1, 0],
                                            lineWidth=2,
                                            lifeTime=0.3)
                    prev_orn0 = orn
                elif agent._name == 'a1':
                    if agent._change == 1:
                        try:
                            orn = orn1[agent._current_index]
                        except:
                            orn = orn1[-1]
                    else :
                        orn = prev_orn1
                    
                    poses = [tuple(pos), orn]

                    while cont == False:
                        #Dummy= True (hand EE)
                        ikRight,_,_ = adam.kinematics.move_arm_to_pose("right", poses,dummy=False)
                        IkR.append(ikRight[2:8])
                        """ link_state = p.getLinkState(adam.robot_id, 29)
                        currentEEPos,ori = link_state[4],link_state[5] """
                        currentEEPos,ori = adam.kinematics.calculate_arm_forward_kinematics("right")
                        """ finalPathR.append(np.array(currentEEPos))
                        finalVelocity.append(np.array(vel)) """
                        
                        diff =np.linalg.norm(np.array(currentEEPos) - np.array(pos))
                        if  diff < 0.05:
                            cont = True
                            pos_D=pos
                            stateD = state
                        else:
                            cont = False
                    print("Derecha:",agent._current_index)
                    p.resetBasePositionAndOrientation(agentId2, pos_D, [0, 0, 0, 1])
                    p.resetBaseVelocity(agentId2, linearVelocity=vel)
                    for s0, s1 in itertools.pairwise(stateD):
                        if s0 is None or s1 is None:
                            break
                        x0, y0, z0, *_ = s0
                        x1, y1, z1, *_ = s1
                        # Dibujar la línea en PyBullet:
                        p.addUserDebugLine([x0/100, y0/100, z0/100], [x1/100, y1/100, z1/100],
                                            lineColorRGB=[0, 0, 1],
                                            lineWidth=2,
                                            lifeTime=0.3)
                    prev_orn1 = orn
            env.step_move()
        
            if adam.useSimulation and adam.useRealTimeSimulation == 0:
                p.stepSimulation()
                time.sleep(adam.t)
            """ elif adam.useRealTimeSimulation:
                time.sleep(0.004) """
        except KeyboardInterrupt:
            """ plot_paths_3d(finalPathR, path_agent1)
            plot_final_velocity(finalVelocity) """
            scipy.io.savemat("datos_Derecha2.mat", {"path_armR": IkR})
            scipy.io.savemat("datos_Izquierda2.mat", {"path_armL": IkL})
            time.sleep(10000)



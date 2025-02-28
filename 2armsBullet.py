import pybullet as p
import pybullet_data
import time
import numpy as np
import itertools
import math

from motionGBP.obstacle3D import ObstacleMap3D
from motionGBP.agent3D import Agent3D, Env

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dt = 1./240.

def generate_curved_path(initial, final, num_points=50):

    x0, y0, z0, *_ = initial
    x1, y1, z1, *_ = final

    control_x = (x0 + x1) / 2
    control_y = min(y0, y1) - 0.5
    control_z = (z0 + z1) / 2

    path = []
    t_values = np.linspace(0, 1, num_points)
    for t in t_values:
        x = (1 - t)**2 * x0 + 2 * (1 - t) * t * control_x + t**2 * x1
        y = (1 - t)**2 * y0 + 2 * (1 - t) * t * control_y + t**2 * y1
        z = (1 - t)**2 * z0 + 2 * (1 - t) * t * control_z + t**2 * z1
        path.append((x, y, z))
    return path

def generate_path(initial, final, num_points=100):

    x0, y0, z0, *_ = initial
    x1, y1, z1, *_ = final

    path = [(x0 + (x1 - x0) * i / (num_points - 1), 
             y0 + (y1 - y0) * i / (num_points - 1),
             z0 + (z1 - z0) * i / (num_points - 1)) for i in range(num_points)]
    return path

def plot_paths_3d(finalPathR, path_agent1_EE):
    finalPathR = np.array(finalPathR)
    path_agent1_EE = np.array(path_agent1_EE)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(finalPathR[:, 0]*100, finalPathR[:, 1]*100, finalPathR[:, 2]*100, label='Agent path', color='red')
    ax.plot(path_agent1_EE[:, 0], path_agent1_EE[:, 1], path_agent1_EE[:, 2], label='LfD path', color='blue')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Agent position using GBP algorithm')
    ax.legend()
    
    plt.show()
def plot_final_velocity(finalVelocity):

    finalVelocity = np.array(finalVelocity)*10  
    time = np.arange(len(finalVelocity))  

    plt.figure(figsize=(8, 5))
    axes = ['X', 'Y', 'Z']
    for i in range(finalVelocity.shape[1]):
        plt.plot(time, finalVelocity[:, i], label=f'Velocity agent in{axes[i]}')

    plt.xlabel('Time (iters)')
    plt.ylabel('Velocity (cm/s)')
    plt.title('Agent velocity using GBP algorithm')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == '__main__':

    physicsClient = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")
    
    # Load Kukas
    
    kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
    p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, -1, 1])
    kukaEndEffectorIndex = 6
    numJoints = p.getNumJoints(kukaId)
    if numJoints != 7:
        exit()

    kukaId2 = p.loadURDF("kuka_iiwa/model.urdf", [0, 0.7, 0], useFixedBase=True)
    p.resetBasePositionAndOrientation(kukaId2, [0, 0.7, 0], [0, 0, 1, 1])
    kukaEndEffectorIndex2 = 6
    numJoints2 = p.getNumJoints(kukaId2)
    if numJoints2 != 7:
        exit()

    
    ll = [ -3.05]*7     
    ul = [ 3.05]*7        
    jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]  
    rp = [0, 0, 0, 0.5 * math.pi, 0, -0.5 * math.pi * 0.66, 0]  
    jd = [0.1] * 7 

    #* Initial position for each arm
    for i in range(numJoints):
        p.resetJointState(kukaId, i, rp[i])
    for i in range(numJoints2):
        p.resetJointState(kukaId2, i, rp[i])
    
    #* --- Fixed obstacles ---
    obstacle_radius = 0.05
    obstacle_position = [0.1, 0.25, 0.5]
    omap = ObstacleMap3D()
    omap2 = ObstacleMap3D()
    omap.set_sphere('', 10, 25, 50, 5)
    omap2.set_sphere('', 10, 25, 50, 5)


    #* --- Initail points ---
    
    initial_position0 = [-45, 15, 30, 10, 20, 0]  
    final_position0   = [30, 40, 70, 10, 20, 5]
    initial_position1 = [30, 45, 35, 10, 20, 5] 
    final_position1   = [-40, 15, 55, 10, 20, 0] 
    #[0.1, 0.25, 0.5] #Obs
    
    
    agent_radius = 0.1

    num_points = 100
    path_agent0 = generate_curved_path(initial_position0, final_position0, num_points=num_points)
    path_agent1 = generate_curved_path(initial_position1, final_position1, num_points=num_points)

    
    agentCollisionShape = p.createCollisionShape(p.GEOM_SPHERE, radius=agent_radius)
    agentVisualShape = p.createVisualShape(p.GEOM_SPHERE, radius=agent_radius, rgbaColor=[0, 1, 0, 0.6])
    agentCollisionShape2 = p.createCollisionShape(p.GEOM_SPHERE, radius=agent_radius)
    agentVisualShape2 = p.createVisualShape(p.GEOM_SPHERE, radius=agent_radius, rgbaColor=[0, 0, 1, 0.6])

    agentId = p.createMultiBody(baseMass=0,
                                baseVisualShapeIndex=agentVisualShape,
                                basePosition=path_agent0[0])
    agentId2 = p.createMultiBody(baseMass=0,
                                baseVisualShapeIndex=agentVisualShape2,
                                basePosition=path_agent1[0])
    
    #* Agents
    agent0 = Agent3D('a0', initial_position0, final_position0, steps=12, radius=10, omap=omap2, path=path_agent0)
    agent1 = Agent3D('a1', initial_position1, final_position1, steps=12, radius=10, omap=omap, path=path_agent1)
    env = Env()
    env.add_agent(agent0)
    env.add_agent(agent1)

    #* --- Trajectories ---
    for i in range(len(path_agent0) - 1):
        start = (path_agent0[i][0]/100, path_agent0[i][1]/100, path_agent0[i][2]/100)
        end = (path_agent0[i+1][0]/100, path_agent0[i+1][1]/100, path_agent0[i+1][2]/100)
        p.addUserDebugLine(start, end, lineColorRGB=[0, 1, 0], lineWidth=2, lifeTime=0)
    for i in range(len(path_agent1) - 1):
        start = (path_agent1[i][0]/100, path_agent1[i][1]/100, path_agent1[i][2]/100)
        end = (path_agent1[i+1][0]/100, path_agent1[i+1][1]/100, path_agent1[i+1][2]/100)
        p.addUserDebugLine(start, end, lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=0)
    
    # * Orientation fixed for the examples
    orn = p.getQuaternionFromEuler([0, math.pi, 0])
    
    #* --- Initial position for each arm ---
    jointPoses = p.calculateInverseKinematics(
        kukaId, 
        kukaEndEffectorIndex, 
        [initial_position0[0]/100, initial_position0[1]/100, initial_position0[2]/100],
        orn, 
        lowerLimits=ll, 
        upperLimits=ul, 
        jointRanges=jr, 
        restPoses=rp, 
        jointDamping=jd)
    jointPoses2 = p.calculateInverseKinematics(
        kukaId2, 
        kukaEndEffectorIndex2, 
        [initial_position1[0]/100, initial_position1[1]/100, initial_position1[2]/100],
        orn, 
        lowerLimits=ll, 
        upperLimits=ul, 
        jointRanges=jr, 
        restPoses=rp, 
        jointDamping=jd)
    
    for _ in range(100):
        for j in range(numJoints):
            p.setJointMotorControl2(
                bodyIndex=kukaId,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=jointPoses[j])
            p.setJointMotorControl2(
                bodyIndex=kukaId2,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=jointPoses2[j])
        p.stepSimulation()
        time.sleep(0.1)
    
    #* === Dynamic Obstacles in each arm ===
    arm1_obstacle_ids = []
    arm1_obstacle_names = []
    sphere_radius = 0.09  # radio mayor
    for i in range(numJoints-1):
        linkState = p.getLinkState(kukaId, i)
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
    for i in range(numJoints2-1):
        linkState = p.getLinkState(kukaId2, i)
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
    
        
    obstacleCollisionShape = p.createCollisionShape(p.GEOM_SPHERE, radius=obstacle_radius)
    obstacleVisualShape = p.createVisualShape(p.GEOM_SPHERE, radius=obstacle_radius, rgbaColor=[1, 0, 0, 0.5])
    obstacleId = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=obstacleCollisionShape,
                                baseVisualShapeIndex=obstacleVisualShape,
                                basePosition=obstacle_position)
    
    
    #* === Main Loop ===
    
    finalPathR = []
    finalPathL = []
    finalVelocityR = []
    finalVelocityL = []
    
    while True:
        try:
            #* Dynamic obstacles (in each arm)
            for i in range(numJoints-1):
                linkState = p.getLinkState(kukaId, i)
                pos_link = linkState[4]
                p.resetBasePositionAndOrientation(arm1_obstacle_ids[i], pos_link, [0, 0, 0, 1])
                
                omap.objects[arm1_obstacle_names[i]]['centerx'] = pos_link[0]*100
                omap.objects[arm1_obstacle_names[i]]['centery'] = pos_link[1]*100
                omap.objects[arm1_obstacle_names[i]]['centerz'] = pos_link[2]*100
            for i in range(numJoints2-1):
                linkState = p.getLinkState(kukaId2, i)
                pos_link = linkState[4]
                p.resetBasePositionAndOrientation(arm2_obstacle_ids[i], pos_link, [0, 0, 0, 1])
                omap2.objects[arm2_obstacle_names[i]]['centerx'] = pos_link[0]*100
                omap2.objects[arm2_obstacle_names[i]]['centery'] = pos_link[1]*100
                omap2.objects[arm2_obstacle_names[i]]['centerz'] = pos_link[2]*100
                
            
            env.step_plan()
            
            for agent in env._agents:
                state = agent.get_state()
                if state[0] is None:
                    continue
                
                x, y, z, vx, vy, vz = state[0]
                pos = [x/100, y/100, z/100]
                vel = [vx/100, vy/100, vz/100]
                
                if agent._name == 'a0':
                    #* --- Control of first arm (kukaId) ---
                    jointPoses = p.calculateInverseKinematics(
                        kukaId, 
                        kukaEndEffectorIndex, 
                        pos, 
                        orn, 
                        lowerLimits=ll, 
                        upperLimits=ul, 
                        jointRanges=jr, 
                        restPoses=rp, 
                        jointDamping=jd,
                        residualThreshold=1e-5)
                    cont = False
                    while not cont:
                        for j in range(numJoints):
                            p.setJointMotorControl2(
                                bodyIndex=kukaId,
                                jointIndex=j,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[j],
                                positionGain=0.5,
                                force=500)
                        p.stepSimulation()
                        time.sleep(0.01)
                        linkState = p.getLinkState(kukaId, kukaEndEffectorIndex)
                        currentEEPos = np.array(linkState[4])
                        diff = np.linalg.norm(currentEEPos - np.array(pos))
                        #print("Brazo 1 - Efector:", currentEEPos, "Objetivo:", pos, "Dif:", diff)
                        if diff < 0.05:
                            cont = True
                            finalPathR.append(np.array(currentEEPos))
                            finalVelocityR.append(np.array(vel))
                    p.resetBasePositionAndOrientation(agentId, pos, [0, 0, 0, 1])
                    p.resetBaseVelocity(agentId, linearVelocity=vel)
                    for s0, s1 in itertools.pairwise(state):
                        if s0 is None or s1 is None:
                            break
                        x0, y0, z0, *_ = s0
                        x1, y1, z1, *_ = s1
                        p.addUserDebugLine([x0/100, y0/100, z0/100], [x1/100, y1/100, z1/100],
                                            lineColorRGB=[0, 1, 0],
                                            lineWidth=2,
                                            lifeTime=0.3)
                
                elif agent._name == 'a1':
                    #* --- Control of second arm (kukaId2) ---
                    jointPoses2 = p.calculateInverseKinematics(
                        kukaId2, 
                        kukaEndEffectorIndex2, 
                        pos, 
                        orn, 
                        lowerLimits=ll, 
                        upperLimits=ul, 
                        jointRanges=jr, 
                        restPoses=rp, 
                        jointDamping=jd,
                        residualThreshold=1e-5)
                    cont2 = False
                    while not cont2:
                        for j in range(numJoints2):
                            p.setJointMotorControl2(
                                bodyIndex=kukaId2,
                                jointIndex=j,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses2[j],
                                positionGain=0.5,
                                force=500)
                        p.stepSimulation()
                        time.sleep(0.01)
                        linkState2 = p.getLinkState(kukaId2, kukaEndEffectorIndex2)
                        currentEEPos2 = np.array(linkState2[4])
                        diff2 = np.linalg.norm(currentEEPos2 - np.array(pos))
                        #print("Brazo 2 - Efector:", currentEEPos2, "Objetivo:", pos, "Dif:", diff2)
                        if diff2 < 0.05:
                            cont2 = True
                            finalPathL.append(np.array(currentEEPos2))
                            finalVelocityL.append(np.array(vel))
                    p.resetBasePositionAndOrientation(agentId2, pos, [0, 0, 0, 1])
                    p.resetBaseVelocity(agentId2, linearVelocity=vel)
                    for s0, s1 in itertools.pairwise(state):
                        if s0 is None or s1 is None:
                            break
                        x0, y0, z0, *_ = s0
                        x1, y1, z1, *_ = s1
                        p.addUserDebugLine([x0/100, y0/100, z0/100], [x1/100, y1/100, z1/100],
                                            lineColorRGB=[0, 0, 1],
                                            lineWidth=2,
                                            lifeTime=0.3)
            
            env.step_move()
            p.stepSimulation()
            time.sleep(0.01)
        except KeyboardInterrupt:
            plot_paths_3d(finalPathL, path_agent1)
            plot_paths_3d(finalPathR, path_agent0)
            plot_final_velocity(finalVelocityR)
            plot_final_velocity(finalVelocityL)
            

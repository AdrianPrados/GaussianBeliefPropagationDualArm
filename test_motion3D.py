import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from motionGBP.obstacle3D import ObstacleMap3D
from motionGBP.agent3D import Agent3D, Env
import time

def generate_curved_path(initial, final, num_points=50):
    x0, y0, z0, _,_,_ = initial
    x1, y1, z1, _,_,_ = final

    control_x = (x0 + x1) / 2
    control_y = min(y0, y1) - 0.5
    control_z = (z0 + z1) / 2  # Punto de control en z

    path = []
    t_values = np.linspace(0, 1, num_points)
    for t in t_values:
        x = (1 - t)**2 * x0 + 2 * (1 - t) * t * control_x + t**2 * x1
        y = (1 - t)**2 * y0 + 2 * (1 - t) * t * control_y + t**2 * y1
        z = (1 - t)**2 * z0 + 2 * (1 - t) * t * control_z + t**2 * z1
        path.append((x, y, z))
    
    return path

def generate_path(initial, final, num_points=100):
    x0, y0, z0, _ = initial
    x1, y1, z1, _ = final

    path = [(x0 + (x1 - x0) * i / (num_points - 1), 
             y0 + (y1 - y0) * i / (num_points - 1),
             z0 + (z1 - z0) * i / (num_points - 1)) for i in range(num_points)]
    
    return path

if __name__ == '__main__':
    
    omap = ObstacleMap3D()
    omap.set_sphere('', 14, 0, 2.5, 5)

    initial_position0 = [0, 0, 0, 10, 20, 0]  
    final_position0 = [35, 30, 10, 10, 20, 5] 

    initial_position1 = [35, 30, 10, 10, 20, 0]  
    final_position1 = [0, 0, 0, 0, 20, 0]
    
    path_agent0 = generate_curved_path(initial_position0, final_position0)
    path_agent1 = generate_curved_path(initial_position1, final_position1)

    agent0 = Agent3D('a0', initial_position0, final_position0, steps=12, radius=5, omap=omap, path=path_agent0)
    agent1 = Agent3D('a1', initial_position1, final_position1, steps=12, radius=5, omap=omap, path=path_agent1)

    env = Env()
    env.add_agent(agent0)
    #env.add_agent(agent1)

    colors = {
        agent0: 'green',
        agent1: 'blue',
    }

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    
    ax.set_xlim3d([0, 400])
    ax.set_ylim3d([0, 400])
    ax.set_zlim3d([0, 400])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    def set_axes_equal(ax):

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        max_range = max(x_range, y_range, z_range)

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
        ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
        ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

    set_axes_equal(ax)

    # Dibujar obstáculos (esferas en 3D)
    for o in omap.objects.values():
        if o['type'] == 'sphere':
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = o['radius'] * np.outer(np.cos(u), np.sin(v)) + o['centerx']
            y = o['radius'] * np.outer(np.sin(u), np.sin(v)) + o['centery']
            z = o['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + o['centerz']
            ax.plot_wireframe(x, y, z, color='red', linewidth=0.5)

    while True:
        
        ax.cla()
        
        
        ax.set_xlim3d([-10, 50])
        ax.set_ylim3d([-20, 20])
        ax.set_zlim3d([-1, 1])
        
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        set_axes_equal(ax)

        
        for o in omap.objects.values():
            if o['type'] == 'sphere':
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x = o['radius'] * np.outer(np.cos(u), np.sin(v)) + o['centerx']
                y = o['radius'] * np.outer(np.sin(u), np.sin(v)) + o['centery']
                z = o['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + o['centerz']
                ax.plot_wireframe(x, y, z, color='red', linewidth=0.5)

        # Dibujar el camino del agente 0
        path_x, path_y, path_z = zip(*path_agent0)
        ax.plot(path_x, path_y, path_z, color='green', linestyle='--', label='Path Agent0')
        
        """ path_x2, path_y2, path_z2 = zip(*path_agent1)
        ax.plot(path_x2, path_y2,path_z2, color='blue', linestyle='--', label='Path Agent2') """
        
        # Planificar y mover agentes

        env.step_plan()
        
        for agent in env._agents:
            #time.sleep(1000)
            color = colors[agent]
            state = agent.get_state()
            if state[0] is None:
                continue

            # Dibujar el agente como una esfera
            x, y, z, _, _, _ = state[0]
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_sphere = agent._radius * np.outer(np.cos(u), np.sin(v)) + x
            y_sphere = agent._radius * np.outer(np.sin(u), np.sin(v)) + y
            z_sphere = agent._radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z
            agent_sphere= ax.plot_surface(x_sphere, y_sphere, z_sphere, color=color, alpha=0.6)
            #ax.add_patch(agent_sphere)

            # Dibujar trayectorias del agente
            for s0, s1 in itertools.pairwise(state):
                if s0 is None or s1 is None:
                    break
                x0, y0, z0, _, _, _ = s0
                x1, y1, z1, _, _, _ = s1
                #print("Valores de s0 y s1: ",s0,s1)
                ax.plot([x0, x1], [y0, y1], [z0, z1], color=color)
            #time.sleep(1000)
        #print(agent0.get_state()[0])
        env.step_move()

        # Actualizar gráfico
        plt.pause(0.01)

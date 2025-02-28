""" import matplotlib.pyplot as plt
import numpy as np
import itertools
from motion.obstacle import ObstacleMap
from motion.agent import Agent, Env
import time
from robotarm import ArmSimulator

def generate_curved_path(initial, final, num_points=50):
    x0, y0, _, _ = initial
    x1, y1, _, _ = final

    # Generar un punto de control intermedio para la curvatura
    control_x = (x0 + x1) / 2
    control_y = min(y0, y1) - 0.5  # Elevar el camino en Y para crear curvatura

    # Crear los puntos del camino usando interpolación cuadrática (Bezier)
    path = []
    t_values = np.linspace(0, 1, num_points)
    for t in t_values:
        x = (1 - t)**2 * x0 + 2 * (1 - t) * t * control_x + t**2 * x1
        y = (1 - t)**2 * y0 + 2 * (1 - t) * t * control_y + t**2 * y1
        path.append((x, y))
    return path

def generate_path(initial, final, num_points=50):
        x0, y0, _, _ = initial
        x1, y1, _, _ = final
        path = []
        for i in range(num_points):
            x = x0 + (x1 - x0) * i / (num_points - 1)
            y = y0 + (y1 - y0) * i / (num_points - 1)
            path.append((x, y))
        return path

if __name__ == '__main__':
    omap = ObstacleMap()
    omap.set_circle('', 145, 0, 40)

    initial_position0 = [0, 0, 10, 20]  
    final_position0 = [350, 100, 10, 20] 

    initial_position1 = [-44,195,0,0]  
    final_position1 = [152, 90, 10, 20]

    path_agent0 = generate_curved_path(initial_position0, final_position0)
    path_agent1 = generate_curved_path(initial_position1, final_position1)

    agent0 = Agent('a0', initial_position0, final_position0, steps=12, radius=10, omap=omap, path=path_agent0)
    agent1 = Agent('a1', initial_position1, final_position1, steps=12, radius=10, omap=omap, path=None)

    env = Env()
    env.add_agent(agent0)
    #env.add_agent(agent1)
    
    # Crear una instancia del simulador de brazo robótico
    arm_simulator = ArmSimulator()

    colors = {
        agent0: 'green',
        agent1: 'blue',
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    # Dibujar los obstáculos
    for o in omap.objects.values():
        if o['type'] == 'circle':
            circle = plt.Circle((o['centerx'], o['centery']), o['radius'], color='red', fill=False, linewidth=2)
            ax.add_patch(circle)

    while True:
        ax.clear()

        # Dibujar los obstáculos
        for o in omap.objects.values():
            if o['type'] == 'circle':
                circle = plt.Circle((o['centerx'], o['centery']), o['radius'], color='red', fill=False, linewidth=2)
                ax.add_patch(circle)
        
        path_x, path_y = zip(*path_agent0)
        ax.plot(path_x, path_y, color='green', linestyle='--', label='Path Agent0')

        env.step_plan()
        for agent in env._agents:
            color = colors[agent]
            state = agent.get_state()
            if state[0] is None:
                continue

            # Dibujar agente
            x, y, vx, vy = state[0]
            agent_circle = plt.Circle((x, y), agent._radius, color=color, fill=False, linewidth=1)
            ax.add_patch(agent_circle)

            # Dibujar camino
            for s0, s1 in itertools.pairwise(state):
                if s0 is None or s1 is None:
                    break
                x0, y0, _, _ = s0
                x1, y1, _, _ = s1
                ax.plot([x0, x1], [y0, y1], color=color)
            # Enviar las posiciones al simulador del brazo robótico si es agent1

            if agent == agent0:
                arm_simulator.run(x, y)
                elbow = arm_simulator.shoulder + arm_simulator.l1 * np.array([np.cos(arm_simulator.theta1), np.sin(arm_simulator.theta1)])
                wrist = elbow + arm_simulator.l2 * np.array([
                    np.sin(np.pi / 2 + arm_simulator.theta2 - arm_simulator.theta1),
                    np.cos(np.pi / 2 + arm_simulator.theta2 - arm_simulator.theta1),
                ])

        env.step_move()
        ax.plot([arm_simulator.shoulder[0], elbow[0], wrist[0]], [arm_simulator.shoulder[1], elbow[1], wrist[1]], 'k-')
        ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'ro')
        ax.plot([arm_simulator.shoulder[0]], [arm_simulator.shoulder[1]], 'bo')

        

        # Configurar límites y mostrar gráfico
        ax.set_xlim(-200, 400)
        ax.set_ylim(-200, 400)
        ax.set_aspect('equal', adjustable='datalim')
        plt.pause(0.01) """
        
import matplotlib.pyplot as plt
import numpy as np
import itertools
from motion.obstacle import ObstacleMap
from motion.agent import Agent, Env
import time
from robotarm import ArmSimulator

def generate_curved_path(initial, final, num_points=50):
    x0, y0, _, _ = initial
    x1, y1, _, _ = final

    control_x = (x0 + x1) / 2
    control_y = min(y0, y1) - 0.5

    path = []
    t_values = np.linspace(0, 1, num_points)
    for t in t_values:
        x = (1 - t)**2 * x0 + 2 * (1 - t) * t * control_x + t**2 * x1
        y = (1 - t)**2 * y0 + 2 * (1 - t) * t * control_y + t**2 * y1
        path.append((x, y))
    return path

def generate_path(initial, final, num_points=50):
    x0, y0, _, _ = initial
    x1, y1, _, _ = final
    path = [(x0 + (x1 - x0) * i / (num_points - 1), y0 + (y1 - y0) * i / (num_points - 1)) for i in range(num_points)]
    return path

if __name__ == '__main__':
    omap = ObstacleMap() #* Agnete 0
    #omap2 = ObstacleMap() #*Agente 1
    #omap2.set_circle('', 145, 0, 70)
    #omap.set_circle('obsReal', 145, 0, 70)
    initial_position0 = [0, 0, 10, 20]  
    final_position0 = [125, 50, 10, 20] 

    initial_position1 = [350, 100, 10, 20]   
    final_position1 = [125, 50, 10, 20]

    path_agent0 = generate_curved_path(initial_position0, final_position0)
    path_agent1 = generate_curved_path(initial_position1, final_position1)

    agent0 = Agent('a0', initial_position0, final_position0, steps=12, radius=10, omap=omap, path=path_agent0)
    agent1 = Agent('a1', initial_position1, final_position1, steps=12, radius=10, omap=omap, path=path_agent1)

    env = Env()
    env.add_agent(agent0)
    env.add_agent(agent1)
    """ arm_simulator = ArmSimulator()
    arm_simulator2 = ArmSimulator()
    arm_simulator2.shoulder = np.array([340, 50]) """
    """ arm_simulator2.theta1 = np.pi/2
    arm_simulator2.theta2 = -(np.pi) """

    colors = {agent0: 'green', agent1: 'blue'}
    fig, ax = plt.subplots(figsize=(10, 8))
    
    while True:
        ax.clear()
        
        # Dibujar los obstáculos
        """ for o in omap2.objects.values():
            if o['type'] == 'circle':
                circle = plt.Circle((o['centerx'], o['centery']), o['radius'], color='purple', fill=False, linewidth=2)
                ax.add_patch(circle) """
                
        # Dibujar el camino del agente 0
        path_x, path_y = zip(*path_agent0)
        ax.plot(path_x, path_y, color='green', linestyle='--', label='Path Agent0')

        env.step_plan()
        
        for agent in env._agents:
            
            color = colors[agent]
            state = agent.get_state()
            if state[0] is None:
                continue
            x, y, vx, vy = state[0]
            
            agent_circle = plt.Circle((x, y), agent._radius, color=color, fill=False, linewidth=1)
            ax.add_patch(agent_circle)
            
            for s0, s1 in itertools.pairwise(state):
                if s0 is None or s1 is None:
                    break
                x0, y0, _, _ = s0
                x1, y1, _, _ = s1
                #print("Valores de s0 y s1: ",s0,s1)
                ax.plot([x0, x1], [y0, y1], color=color)
            #time.sleep(1000)
            """ if agent == agent0:
                success, theta1,theta2= arm_simulator.run(x, y)
                wrist,elbow = arm_simulator.forward_kinematics(theta1=theta1, theta2=theta2)
                elbow = arm_simulator.shoulder + arm_simulator.l1 * np.array([np.cos(arm_simulator.theta1), np.sin(arm_simulator.theta1)])
                wrist = elbow + arm_simulator.l2 * np.array([
                    np.sin(np.pi / 2 + arm_simulator.theta2 - arm_simulator.theta1),
                    np.cos(np.pi / 2 + arm_simulator.theta2 - arm_simulator.theta1),
                ]) """
            """ elif agent == agent1:
                arm_simulator2.run(x, y)
                success, theta1,theta2= arm_simulator2.run(x, y)
                wrist2,elbow2 = arm_simulator.forward_kinematics(theta1=theta1, theta2=theta2)
                elbow2 = arm_simulator2.shoulder + arm_simulator2.l1 * np.array([np.cos(arm_simulator2.theta1), np.sin(arm_simulator2.theta1)])
                time.sleep(1000)
                wrist2 = elbow2 + arm_simulator2.l2 * np.array([
                    np.cos(arm_simulator2.theta1 + arm_simulator2.theta2),
                    np.sin(arm_simulator2.theta1 + arm_simulator2.theta2),
                ]) """
                
                
        # Actualizar la posición de los elementos del brazo robótico
        """ obstacle_positions = np.linspace(elbow, wrist, 15)[1:-1]
        obstacle_positions2 = np.linspace(arm_simulator.shoulder, elbow, 8)[1:-1]
        omap.objects.clear()
        #omap.set_circle('obsReal', 145, 0, 70)
        for i, (ox, oy) in enumerate(obstacle_positions):
            omap.set_circle(f'arm_{i}', ox, oy, 20)
        for i, (ox, oy) in enumerate(obstacle_positions2):
            omap.set_circle(f'shoulder_{i}', ox, oy, 20) """
            
        
        # Actualizar la posición de los elementos del brazo robótico
        """ obstacle_positions22 = np.linspace(elbow2, wrist2, 15)[1:-1]
        obstacle_positions222 = np.linspace(arm_simulator2.shoulder, elbow2, 8)[1:-1]
        omap2.objects.clear()
        #omap.set_circle('obsReal', 145, 0, 70)
        for i, (ox, oy) in enumerate(obstacle_positions22):
            omap2.set_circle(f'arm_{i}', ox, oy, 20)
        for i, (ox, oy) in enumerate(obstacle_positions222):
            omap2.set_circle(f'shoulder_{i}', ox, oy, 20) """
        
        env.step_move()
        
        #print("Lista de obstáculos:", omap.objects.values())
        """ ax.plot([arm_simulator.shoulder[0], elbow[0], wrist[0]], [arm_simulator.shoulder[1], elbow[1], wrist[1]], 'k-')
        ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'ro')
        ax.plot([arm_simulator.shoulder[0]], [arm_simulator.shoulder[1]], 'bo')
        
        ax.plot([arm_simulator2.shoulder[0], elbow2[0], wrist2[0]], [arm_simulator2.shoulder[1], elbow2[1], wrist2[1]], 'k-')
        ax.plot([elbow2[0], wrist2[0]], [elbow2[1], wrist2[1]], 'ro')
        ax.plot([arm_simulator2.shoulder[0]], [arm_simulator2.shoulder[1]], 'bo')
        
        print("Valor wrist 1:", wrist2) """
        print("valor agente 1:", agent1.get_state()[0])
        
        # Dibujar los obstáculos
        for o in omap.objects.values():
            if o['type'] == 'circle':
                circle = plt.Circle((o['centerx'], o['centery']), o['radius'], color='red', fill=False, linewidth=2)
                ax.add_patch(circle)
        
        """ for o in omap2.objects.values():
            if o['type'] == 'circle':
                circle = plt.Circle((o['centerx'], o['centery']), o['radius'], color='red', fill=False, linewidth=2)
                ax.add_patch(circle) """
        
        ax.set_xlim(-200, 400)
        ax.set_ylim(-200, 400)
        ax.set_aspect('equal', adjustable='datalim')
        plt.pause(0.01)
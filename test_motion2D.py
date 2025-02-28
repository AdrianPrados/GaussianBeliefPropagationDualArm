import pygame as pg
import pygame.locals as pgl
import sys
import itertools
import numpy as np

from motionGBP.obstacle import ObstacleMap
from motionGBP.agent import Agent, Env

def generate_curved_path(initial, final, num_points=200):
    x0, y0, _, _ = initial
    x1, y1, _, _ = final
    
    control_x = (x0 + x1) / 2
    control_y = min(y0, y1) - 50  
    
    path = []
    t_values = np.linspace(0, 1, num_points)
    for t in t_values:
        x = (1 - t)**2 * x0 + 2 * (1 - t) * t * control_x + t**2 * x1
        y = (1 - t)**2 * y0 + 2 * (1 - t) * t * control_y + t**2 * y1
        path.append((x, y))
    return path

""" def generate_path(initial, final, num_points=50):
        x0, y0, _, _ = initial
        x1, y1, _, _ = final
        path = []
        for i in range(num_points):
            x = x0 + (x1 - x0) * i / (num_points - 1)
            y = y0 + (y1 - y0) * i / (num_points - 1)
            path.append((x, y))
        return path """

if __name__ == '__main__':
    omap = ObstacleMap()
    omap.set_circle('', 145, 85, 50)
    
    initial_position0 = [0, 0, 10, 20]  
    final_position0 = [350, 100, 10, 20] 

    initial_position1 = [350, 100, 0, 0]  
    final_position1 = [0, 0, 0, 0]

    """ initial_position0 = [0, 0, 10, 20]  
    final_position0 = [-120, -300, 0, 0] 

    initial_position1 = [350, 100, 0, 0]  
    final_position1 = [0, 0, 0, 0]
    
    initial_position2 = [-120, -300, 0, 0]  
    final_position2 = [0, 0, 0, 0] """

    path_agent0 = generate_curved_path(initial_position0, final_position0)
    path_agent1 = generate_curved_path(initial_position1, final_position1)
    #path_agent2 = generate_curved_path(initial_position2, final_position2)

    # Init
    agent0 = Agent('a0', initial_position0, final_position0, steps=12, radius=15, omap=omap, path=path_agent0)
    agent1 = Agent('a1', initial_position1, final_position1, steps=12, radius=15, omap=omap, path=path_agent1)
    #agent2 = Agent('a2', initial_position2, final_position2, steps=12, radius=15, omap=omap, path=path_agent2)
    env = Env()
    env.add_agent(agent0)
    #env.add_agent(agent1)
    #env.add_agent(agent2)

    colors = {
        agent0: (0, 222, 0),
        agent1: (0, 0, 255),
        #agent2: (255, 0, 255),
    }

    pg.init()
    surf = pg.display.set_mode((1000, 800))
    xoff, yoff = 500, 500

    while True:
        surf.fill((255, 255, 255))

        # Draw obstacles
        for o in omap.objects.values():
            if o['type'] == 'circle':
                pg.draw.circle(surf, (222, 0, 0), (o['centerx']+xoff, o['centery']+yoff), o['radius'], 5)

        env.step_plan()
        for agent in env._agents:
            color = colors[agent]
            state = agent.get_state()
            if state[0] is None:
                continue
            x, y, vx, vy = state[0]
            pg.draw.circle(surf, color, (x+xoff, y+yoff), agent._radius, 1)

            for s0, s1 in itertools.pairwise(state):
                if s0 is None or s1 is None:
                    break
                x0, y0, _, _ = s0
                x1, y1, _, _ = s1
                pg.draw.line(surf, color, (x0+xoff, y0+yoff), (x1+xoff, y1+yoff))
        env.step_move()

        for event in pg.event.get():
            if event.type == pgl.QUIT:
                pg.quit()
                sys.exit()
        pg.display.update()
        #pg.time.wait(1)
        pg.time.delay(1)

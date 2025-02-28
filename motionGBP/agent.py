from typing import Tuple, List, Dict
import numpy as np
from factor_graphs.gaussian import Gaussian
from factor_graphs.factor_graph import VNode, FNode, FactorGraph

from .obstacle import ObstacleMap
from .nodes import DynaFNode, ObstacleFNode, DistFNode, RemoteVNode
import time


class Agent:
    def __init__(
            self, name: str, state, target = None, steps: int = 8, radius: int = 5, omap: ObstacleMap = None, env: 'Env' = None,
            start_position_precision = 100,
            start_velocity_precision = 100,
            target_position_precision = 5,
            target_velocity_precision = 5,
            dynamic_position_precision = 10,
            dynamic_velocity_precision = 2,
            obstacle_precision = 50,
            distance_precision = 1, #! Modificado de 100 a 1
            dt: float = 0.1,
            path = None
        ) -> None:
        assert steps > 1
        if np.shape(state) == ():
            state = np.array([[state]])
        elif len(np.shape(state)) == 1:
            state = np.array(state)[:, None]
        if np.shape(target) == ():
            target = np.array([[target]])
        elif len(np.shape(target)) == 1:
            target = np.array(target)[:, None]

        self._steps = steps
        self._name = name
        self._state = np.array(state)
        self._omap = omap
        self._radius = radius
        self._env = env
        self.path = path if path is not None else []
        self.recovery_point= None
        self.specific_point = None

        self._dt = dt
        self._startFNode_pos_prec = start_position_precision
        self._startFNode_vel_prec = start_velocity_precision
        self._targetFNode_pos_prec = target_position_precision
        self._targetFNode_vel_prec = target_velocity_precision
        self._distFNode_prec = distance_precision
        self._change = 1 #* Tres estados diferentes 1(seguir path), 2(dynamic), 3(reconfigurar), 10(caso de seguir la IK del codo), 11(caso de esquivar obstaculo)

        # Create VNodes
        self._vnodes = [VNode(f'v{i}', [f'v{i}.x', f'v{i}.y', f'v{i}.vx', f'v{i}.vy']) for i in range(steps)]


        #! hasta aqui estan incializados a cero
        # Create FNode for start and target
        self._fnode_start = FNode('fstart', [self._vnodes[0]])
        self._fnode_end = FNode('fend', [self._vnodes[-1]])
        self.set_state(state) #* Actualizacion de los valores del estado de cada nodo
        self.set_target(target)
        self._current_index = 0
        self._target = target

        # Create DynaFNode
        self._fnodes_dyna = [DynaFNode(
            f'fd{i}{i+1}', [self._vnodes[i], self._vnodes[i+1]], dt=self._dt,
            pos_prec=dynamic_position_precision, vel_prec=dynamic_velocity_precision
        ) for i in range(steps-1)]
        # Create ObstacleFNode
        
        self._fnodes_obst = [ObstacleFNode(
            f'fo{i}', [self._vnodes[i]], omap=omap, safe_dist=self.r*1.5,
            z_precision=obstacle_precision
        ) for i in range(1, steps)]
        
        self._graph = FactorGraph()
        
        self._graph.connect(self._vnodes[0], self._fnode_start)
        for v, f in zip(self._vnodes[:-1], self._fnodes_dyna):
            self._graph.connect(v, f)
        for v, f in zip(self._vnodes[1:], self._fnodes_dyna):
            self._graph.connect(v, f)
        for v, f in zip(self._vnodes[1:], self._fnodes_obst):
            self._graph.connect(v, f)
        self._graph.connect(self._vnodes[-1], self._fnode_end)

        self._others = {}

    def __str__(self) -> str:
        return f'({self._name} s={self._state})'

    @property
    def name(self) -> str:
        return self._name
    @property
    def x(self) -> float:
        '''current x'''
        return self._state[0, 0]
    @property
    def y(self) -> float:
        '''current y'''
        return self._state[1, 0]
    @property
    def r(self) -> float:
        '''radius'''
        return self._radius

    def get_state(self) -> List[np.ndarray]:
        poss = []
        for v in self._vnodes:
            if v.belief is None:
                poss.append(None)
            else:
                poss.append(v.belief.mean[:, 0])
        return poss

    def get_target(self) -> np.ndarray:
        return self._fnode_end._factor.mean

    def step_connect(self):
        # Search near agents
        others = self._env.find_near(self)
        for o in others:
            self.setup_com(o)
        for on in list(self._others.keys()):
            if self._others[on] not in others:
                self.end_com(o)

    def step_com(self):
        for o in self._others:
            self.send(o)

    def step_propagate(self):
        self._graph.loopy_propagate()

    """ def step_move(self):
        
        self._fnode_start._factor = Gaussian(self._vnodes[0].dims, self._vnodes[1].mean, self._vnodes[1].belief.cov)
        for i in range(0, self._steps-1):
            v, v_ = self._vnodes[i], self._vnodes[i+1]
            v._belief = Gaussian(v.dims, v_.mean, v_.belief.cov)
            #print("Soy un believer",v._belief)
        s = v_.mean + np.random.rand(4, 1) * 0.01
        s[:2] += s[2:] * self._dt
        print("posicion 2D",s)
        v_._belief = Gaussian(v_.dims, s, v_.belief.cov) """
    #* Function to move the agent and take into account different states
    def step_move(self):
        path = self.path
        #print("Path",path)

        if self._current_index == self._steps:
            actual = path[self._current_index-1]
            if len(actual) == 2:
                actual = (*actual, 0, 0)  # Agregamos velocidad 0 para completar los 4 elementos
            self.set_state(self._state)
            self._current_index = self._current_index +1
            #time.sleep(1000)
        
        #print("Punto desde el que empiezo a calcualr",self._fnode_start._factor)
        other = self._env.find_near(self, range= 40)
        print(self.get_state()[0][:2])
        print("----------------------------")
        #print("Obstaculos:",self._omap.objects)
        
        distObs,_,_=self._omap.get_d_grad(x=self.get_state()[0][:2][0],y=self.get_state()[0][:2][1])
        print("OBSTACLEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE:",distObs)
        print("Esatdo en el que estoy",self.get_state()[0][:2])
        #time.sleep(1000)
        
        #self._fnodes_obst.update_factor()
        print("Otros agentes cercanos",other)
        if not other : #? Casos sin agentes cercanos
            #print("Valor de CHANGEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",self._change)
            if self._change == 3 and distObs > 35: #* Caso busqueda de vuelta
                print("Searching recovery point...")
                d_agente = np.linalg.norm(self.get_state()[0][:2] - np.array(self._fnode_end._factor.mean[:2]).flatten())
                for i in range(len(path)- self._current_index):
                    d = np.linalg.norm(np.array(self._fnode_end._factor.mean[:2]).flatten() - path[self._current_index+i])
                    distRecov,_,_=self._omap.get_d_grad(x=path[self._current_index+i][0],y=path[self._current_index+i][1])
                    if d < d_agente and distRecov > 20:
                        print("Entroooo")
                        try:
                            target_position = path[self._current_index+i+5]
                        except:
                            target_position = path[-1]
                        if len(target_position) == 2:
                            target_position = (*target_position, 0, 0)
                        self.recovery_point = np.array(target_position).reshape(4, 1)
                        self.set_target(self.recovery_point)
                        self._current_index = self._current_index + i+ 5
                        self._change = 2
                        break
                #! Esta aprte se ha añadido para el caso de que existan obstaculos en el camin ode vuelta se fuerza a que vaya al final
                try:
                    print(target_position)
                except:
                    target_position = path[-1]
                    if len(target_position) == 2:
                            target_position = (*target_position, 0, 0)
                    self.recovery_point = np.array(target_position).reshape(4, 1)
                    self.set_target(self.recovery_point)
                self._change=2


            elif self._change == 2 and distObs > 35: #* Caso modificar el punto final al que llegar
                self._fnode_start._factor = Gaussian(self._vnodes[0].dims, self._vnodes[1].mean, self._vnodes[1].belief.cov)
                print("Recovery position")
                
                for i in range(0, self._steps - 1):
                    v, v_ = self._vnodes[i], self._vnodes[i + 1]
                    v._belief = Gaussian(v.dims, v_.mean, v_.belief.cov)
                    
                s = v_.mean + np.random.rand(4, 1) * 0.01
                s[:2] += s[2:] * self._dt
                v._belief = Gaussian(self._vnodes[-1].dims, s, self._vnodes[-1].belief.cov)
                if np.linalg.norm(self.get_state()[0][:2] - np.array(self.recovery_point[:2]).flatten()) <= 8:
                    print("I arrived to the point")
                    self._change = 1
                    self.set_target(self._target)
                else:
                    self._change = 2
                    print("Not there yet")
                
            elif self._change == 1 and distObs > 35: #* Caso seguir el path
                print("Following path")
                    
                try:
                    target_position = path[self._current_index]
                except:
                    target_position = path[-1]
                    if np.linalg.norm(self.get_state()[0][:2] - target_position) <= 8:
                        print("End of the path")
                        exit
                if len(target_position) == 2:
                    target_position = (*target_position, 0, 0)
                
                s = np.array(target_position).reshape(4, 1)
                self._fnode_start._factor = Gaussian(self._vnodes[0].dims, s, self._vnodes[1].belief.cov*0.1)
                s[:2] += s[2:] * self._dt
                v = self._vnodes[1]
                v._belief._info = s
                v._belief._prec= v._belief._prec
                self._current_index = self._current_index + 1
            else:
                self._fnode_start._factor = Gaussian(self._vnodes[0].dims, self._vnodes[1].mean, self._vnodes[1].belief.cov*0.1)
                print("Obstacle when following or recovering")
                
                for i in range(0, self._steps - 1):
                    v, v_ = self._vnodes[i], self._vnodes[i + 1]
                    v._belief = Gaussian(v.dims, v_.mean, v_.belief.cov)
                
                s = v_.mean + np.random.rand(4, 1) * 1
                s[:2] += s[2:] * self._dt
                
                v._belief = Gaussian(self._vnodes[-1].dims, s, self._vnodes[-1].belief.cov)
                self._change = 3
            
        else:
            self._fnode_start._factor = Gaussian(self._vnodes[0].dims, self._vnodes[1].mean, self._vnodes[1].belief.cov*0.1)
            print("Esquiva esquiva")
            
            for i in range(0, self._steps - 1):
                v, v_ = self._vnodes[i], self._vnodes[i + 1]
                v._belief = Gaussian(v.dims, v_.mean, v_.belief.cov)
                
            s = v_.mean + np.random.rand(4, 1) * 1
            s[:2] += s[2:] * self._dt
            v._belief = Gaussian(self._vnodes[-1].dims, s, self._vnodes[-1].belief.cov)
            self._change = 3

        print("-------------------")

    def set_state(self, state):
        
        self._state = np.array(state)
        v0 = self._vnodes[0]
        #print("Los valroes v0",v0.mean)
        cov = np.diag([1/self._startFNode_pos_prec, 1/self._startFNode_pos_prec, 1/self._startFNode_vel_prec, 1/self._startFNode_vel_prec])
        self._fnode_start._factor = Gaussian(v0.dims, state, cov)
        v0._belief = Gaussian(v0.dims, state, cov.copy())
        

        #* Init position for each vnode, to avoid 0 distance in DistFNode
        if self.path !=[]:
            for i in range(1, self._steps):
                state_at_i = self.path[i] 
                if len(state_at_i) == 2:
                    state_at_i = (*state_at_i, 10, 10)
                s = np.array(state_at_i).reshape(4, 1)
                #s[:2] += s[2:] * self._dt
                v = self._vnodes[i]
                v._belief = Gaussian(v.dims, s, cov.copy())
        else:
            for i in range(1, self._steps):
                s = state + np.random.rand(4, 1) * 1
                s[:2] += s[2:] * self._dt
                v = self._vnodes[i]
                v._belief = Gaussian(v.dims, s, cov.copy())
        """ for i in range(1, self._steps):
            s = state + np.random.rand(4, 1) * 0.01
            s[:2] += s[2:] * self._dt
            v = self._vnodes[i]
            v._belief = Gaussian(v.dims, s, cov.copy()) """
        #time.sleep(1000)

    def set_target(self, target):
        if target is not None:
            self._fnode_end._factor = Gaussian(self._vnodes[-1].dims, target, np.diag([1]*4))
        else:
            self._fnode_end._factor = Gaussian.identity(self._vnodes[-1].dims)

    def push_msg(self, msg):
        '''Called by other agent to simulate the other sending message to self agent.'''
        _type, aname, vname, p = msg
        if p is None:
            return
        if aname not in self._others:
            print(f'push msg: name {aname} not found')
            return
        vnodes: List[RemoteVNode] = self._others[aname]['v']
        vnode: RemoteVNode = None
        for v in vnodes:
            if v.name == vname:
                vnode = v
                break
        if vnode is None:
            print('vname not found')
            return

        p: Gaussian
        p._dims = vnode.dims
        if _type == 'belief':
            vnode._belief = p
        if _type == 'f2v': # TODO FIXME dims
            e = vnode.edges[0]
            e.set_message_from(e.get_other(vnode), p)
        if _type == 'v2f': # TODO FIXME dims
            e = vnode.edges[0]
            vnode._msgs[e] = p
            #e.set_message_from(vnode, p)

    def setup_com(self, other: 'Agent'):
        on = other._name
        if on in self._others:
            return

        vnodes = [RemoteVNode(f'{on}.v{i}', [f'{on}.v{i}.x', f'{on}.v{i}.y', f'{on}.v{i}.vx', f'{on}.v{i}.vy']) for i in range(1, self._steps)]
        fnodes = [DistFNode(
            f'{on}.f{i}', [vnodes[i-1], self._vnodes[i]], safe_dist=(self.r+other.r) * 2,
            z_precision=self._distFNode_prec
        ) for i in range(1, self._steps)]

        for i in range(1, self._steps):
            self._graph.connect(self._vnodes[i], fnodes[i-1])
            self._graph.connect(vnodes[i-1], fnodes[i-1])
        self._others[on] = {'a': other, 'v': vnodes, 'f': fnodes}

        other.setup_com(self)

        self.send(on)

    def send(self, name: str):
        other = self._others[name]['a']
        for i in range(1, self._steps):
            vname = f'{self._name}.v{i}'
            v = self._vnodes[i]
            f: FNode = self._others[name]['f'][i-1]

            belief = v.belief.copy()
            other.push_msg(('belief', self._name, vname, belief))

            f2v = f.edges[0].get_message_to(v)
            if f2v is not None:
                f2v = f2v.copy()
            other.push_msg(('f2v', self._name, vname, f2v))

            v2f = f.edges[0].get_message_to(f)
            if v2f is not None:
                v2f = v2f.copy()
            other.push_msg(('v2f', self._name, vname, v2f))

    def end_com(self, name: str):
        if name not in self._others:
            return

        vnodes = self._others[name]['v']
        fnodes = self._others[name]['f']
        for v in vnodes:
            self._graph.remove_node(v)
        for f in fnodes:
            self._graph.remove_node(f)
        other_dict = self._others.pop(name)
        other_dict['a'].end_com(self._name)


class Env:
    def __init__(self) -> None:
        self._agents: List[Agent] = []

    def add_agent(self, a: Agent):
        if a not in self._agents:
            a._env = self
            self._agents.append(a)

    def find_near(self, this: Agent, range: float = 1000, max_num: int = -1) -> List[Agent]:
        agent_ds = []
        for a in self._agents:
            if a is this:
                continue
            #d = np.sqrt((a.x-this.x)**2 + (a.y-this.y)**2)
            d = np.linalg.norm(a.get_state()[0][:2] - this.get_state()[0][:2])
            if d < range:
                agent_ds.append((a, d))
        agent_ds.sort(key=lambda ad: ad[1])
        if max_num > 0:
            agent_ds = agent_ds[:max_num]
        return [a for a, d in agent_ds]

    def step_plan(self, iters = 12): #* por defecto tiene 12
        for a in self._agents:
            a.step_connect()
        for i in range(iters):
            for a in self._agents:
                a.step_com()
            for a in self._agents:
                a.step_propagate()
            """ if i == 2:
                time.sleep(10000) """

    def step_move(self):
        for a in self._agents:
            a.step_move()
            

from typing import Tuple, List, Dict
import numpy as np
from factor_graphs.gaussian import Gaussian
from factor_graphs.factor_graph import VNode, FNode, FactorGraph

from .obstacle3D import ObstacleMap3D
import time


class RemoteVNode(VNode):
    def __init__(self, name: str, dims: list, belief: Gaussian = None) -> None:
        super().__init__(name, dims, belief)
        self._msgs = {}

    def update_belief(self) -> Gaussian:
        return None

    def calc_msg(self, edge):
        return self._msgs.get(edge, None)


class DynaFNode(FNode):
    def __init__(self, name: str, vnodes: List[VNode], factor: Gaussian = None,
                dt: float = 0.1, pos_prec: float = 10, vel_prec: float = 2) -> None:
        assert len(vnodes) == 2
        super().__init__(name, vnodes, factor)
        self._dt = dt
        self._pos_prec = pos_prec
        self._vel_prec = vel_prec

    def update_factor(self):
        #print("Entro1")
        dt = self._dt
        v0 = self._vnodes[0].mean
        v1 = self._vnodes[1].mean
        """ print("V1",v1)
        print("V0",v0)
        time.sleep(2) """
        v = np.vstack([v0, v1])  # [12, 1] for 3D (x, y, z, vx, vy, vz)
        z = np.zeros((6, 1))  # Observing position and velocity (x, y, z, vx, vy, vz)

        # Kinetic model for 3D
        k = np.identity(6)   # [6, 6]
        k[:3, 3:] = np.identity(3) * dt

        h = k @ v0 - v1  # [6, 1]
        # Jacobian of h
        """ jacob = np.array([
            [1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0],  # h(x)[0] = dx = x(k) + vx(k) * dt - x(k+1)
            [0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0],  # h(x)[1] = dy = y(k) + vy(k) * dt - y(k+1)
            [0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0],  # h(x)[2] = dz = z(k) + vz(k) * dt - z(k+1)
            [0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0],  # h(x)[3] = dvx = vx(k) - vx(k+1)
            [0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0],  # h(x)[4] = dvy = vy(k) - vy(k+1)
            [0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0],  # h(x)[5] = dvz = vz(k) - vz(k+1)
        ]) """  # [6, 12]
        
        jacob = np.array([
            [1, 0, 0,  dt, 0,  0, -1, 0,  0,  0,  0,  0],  # h(x)[0]: x(k) + vx(k)*dt - x(k+1)
            [0, 1, 0,  0,  dt, 0,  0, -1,  0,  0,  0,  0],  # h(x)[1]: y(k) + vy(k)*dt - y(k+1)
            [0, 0, 1,  0,  0,  dt, 0,  0, -1,  0,  0,  0],  # h(x)[2]: z(k) + vz(k)*dt - z(k+1)
            [0, 0, 0,  1,  0,  0,  0,  0,  0, -1,  0,  0],  # h(x)[3]: vx(k) - vx(k+1)
            [0, 0, 0,  0,  1,  0,  0,  0,  0,  0, -1,  0],  # h(x)[4]: vy(k) - vy(k+1)
            [0, 0, 0,  0,  0,  1,  0,  0,  0,  0,  0, -1]   # h(x)[5]: vz(k) - vz(k+1)
        ])  # [6, 12]   

        precision = np.diag([self._pos_prec, self._pos_prec, self._pos_prec, 
                            self._vel_prec, self._vel_prec, self._vel_prec])  # Precision matrix

        prec = jacob.T @ precision @ jacob
        info = jacob.T @ precision @ (jacob @ v + z - h)

        self._factor = Gaussian.from_info(self.dims, info, prec)


class ObstacleFNode(FNode):
    def __init__(self, name: str, vnodes: List[VNode], factor: Gaussian = None,
                omap: ObstacleMap3D = None, safe_dist: float = 5, z_precision: float = 100) -> None:
        assert len(vnodes) == 1
        super().__init__(name, vnodes, factor)
        self._omap = omap
        self._safe_dist = safe_dist
        self._z_precision = z_precision

    def update_factor(self):
        #print("Entro2")
        # target: ||h(x) - z)||2 -> 0
        z = np.zeros((1, 1))
        v = self._vnodes[0].mean  # [6, 1] for 3D (x, y, z, vx, vy, vz)

        # Using the 3D obstacle map
        distance, gradx, grady, gradz = self._omap.get_d_grad(v[0, 0], v[1, 0], v[2, 0])

        #print(f"distance: {distance}, gradx: {gradx}, grady: {grady}, gradz: {gradz}")

        h = np.array([[max(0, 1 - distance / self._safe_dist)]])
        
        
        if distance > self._safe_dist:
            jacob = np.zeros((1, 6))  # Jacobian for 3D
        else:
            jacob = np.array([[-gradx/self._safe_dist, -grady/self._safe_dist, -gradz/self._safe_dist, 0, 0, 0]])  # [1, 6]

        precision = np.identity(1) * self._z_precision * self._safe_dist**2

        prec = jacob.T @ precision @ jacob
        info = jacob.T @ precision @ (jacob @ v + z - h)
        
        
        self._factor = Gaussian.from_info(self.dims, info, prec)


""" class DistFNode(FNode):
    def __init__(self, name: str, vnodes: List[VNode], factor: Gaussian = None,
                safe_dist: float = 20, z_precision: float = 100) -> None:
        assert len(vnodes) == 2
        super().__init__(name, vnodes, factor)
        self._safe_dist = safe_dist
        self._z_precision = z_precision

    def update_factor(self):
        print("Entro3")
        z = np.zeros((1, 1))
        v0 = self._vnodes[0].mean  # [6, 1] for 3D
        v1 = self._vnodes[1].mean  # [6, 1] for 3D
        if np.allclose(v0, v1):
            v1 += np.random.rand(6, 1) * 0.01
        v = np.vstack([v0, v1])  # [12, 1]

        distance = np.linalg.norm(v0[:3, 0] - v1[:3, 0])  # Using only the position part
        distance_gradx0, distance_grady0, distance_gradz0 = (v0[0, 0] - v1[0, 0], 
                                                            v0[1, 0] - v1[1, 0], 
                                                            v0[2, 0] - v1[2, 0])
        distance /= np.linalg.norm([distance_gradx0, distance_grady0, distance_gradz0])  # Normalize distance

        if distance > self._safe_dist:
            prec = np.identity(12) * 0.0001
            info = prec @ v
        else:
            h = np.array([[1 - distance / self._safe_dist]])
            jacob = np.array([[
                -distance_gradx0/self._safe_dist, -distance_grady0/self._safe_dist, -distance_gradz0/self._safe_dist, 
                0, 0, 0,
                distance_gradx0/self._safe_dist, distance_grady0/self._safe_dist, distance_gradz0/self._safe_dist,
                0, 0, 0]])  # [1, 12]
            
            precision = np.identity(1) * self._z_precision * (self._safe_dist**2)

            prec = jacob.T @ precision @ jacob
            info = jacob.T @ precision @ (jacob @ v + z - h)

        self._factor = Gaussian.from_info(self.dims, info, prec) """
        
class DistFNode(FNode):
    def __init__(self, name: str, vnodes: List[VNode], factor: Gaussian = None,
                safe_dist: float = 20, z_precision: float = 100) -> None:
        assert len(vnodes) == 2
        super().__init__(name, vnodes, factor)
        self._safe_dist = safe_dist
        self._z_precision = z_precision

    def update_factor(self):
        #print("Entro3")
        # target: ||h(x) - z||2 -> 0
        z = np.zeros((1, 1))
        v0 = self._vnodes[0].mean  # [6, 1] in 3D
        v1 = self._vnodes[1].mean  # [6, 1] in 3D

        if np.allclose(v0, v1):
            v1 += np.random.rand(6, 1) * 0.01

        v = np.vstack([v0, v1])  # [12, 1]
        distance = np.linalg.norm(v0[:3, 0] - v1[:3, 0])  # 3D distance
        distance_grad = (v0[:3, 0] - v1[:3, 0]) / distance  # Gradient in 3D

        if distance > self._safe_dist:
            
            prec = np.identity(12) * 0.0001
            info = prec @ v

        else:
            h = np.array([[1 - distance / self._safe_dist]])
            jacob = np.zeros((1, 12))
            jacob[0, :3] = -distance_grad / self._safe_dist
            jacob[0, 6:9] = distance_grad / self._safe_dist

            precision = np.identity(1) * self._z_precision * (self._safe_dist ** 2)

            prec = jacob.T @ precision @ jacob
            info = jacob.T @ precision @ (jacob @ v + z - h)

            # NOTE
            # Adjust prec to make it invertible while preserving the structure.
            prec[3:6, 3:6] = np.identity(3)
            prec[9:12, 9:12] = np.identity(3)
            #print("Prec", prec)
            #time.sleep(10000)

        self._factor = Gaussian.from_info(self.dims, info, prec)

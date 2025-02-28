from typing import Tuple, Dict
import numpy as np
import time


class ObstacleMap3D:
    def __init__(self) -> None:
        self.objects = {}

    def set_sphere(self, name: str, centerx: float, centery: float, centerz: float, radius: float) -> None:
        """
        Define a spherical obstacle in the 3D space.
        """
        o = {
            'type': 'sphere',
            'name': name,
            'centerx': centerx,
            'centery': centery,
            'centerz': centerz,
            'radius': radius
        }
        self.objects[name] = o

    def get_d_grad(self, x: float, y: float, z: float) -> Tuple[float, float, float, float]:
        """
        Computes the distance to the nearest obstacle and the gradient to avoid it.
        """
        mindist = np.inf
        mino = None

        for o in self.objects.values():
            if o['type'] == 'sphere':
                ox, oy, oz, r = o['centerx'], o['centery'], o['centerz'], o['radius']  
                d = np.sqrt((x - ox) ** 2 + (y - oy) ** 2 + (z - oz) ** 2) - r
                if d < mindist:
                    mindist = d
                    mino = o

        if mino is None:
            return np.inf, 0, 0, 0

        if mino['type'] == 'sphere':
            ox, oy, oz = mino['centerx'], mino['centery'], mino['centerz']
            #ox, oy ,oz= o['centerx'], o['centery'], o['centerz']
            dx, dy, dz = x - ox, y - oy, z - oz
            mag = np.sqrt(dx**2 + dy**2 + dz**2)
            return mindist, dx/mag, dy/mag, dz/mag


# Ejemplo de uso
if __name__ == "__main__":
    # Crear un mapa 3D de obstáculos
    obstacle_map = ObstacleMap3D()

    # Agregar esferas al mapa
    obstacle_map.set_sphere("sphere1", centerx=1.0, centery=2.0, centerz=3.0, radius=1.0)
    obstacle_map.set_sphere("sphere2", centerx=-1.0, centery=0.0, centerz=2.0, radius=0.5)

    # Calcular distancia y gradiente desde un punto dado
    point_x, point_y, point_z = 0.0, 1.0, 2.0
    dist, gradx, grady, gradz = obstacle_map.get_d_grad(point_x, point_y, point_z)

    print(f"Distance to nearest obstacle: {dist}")
    print(f"Gradient to avoid obstacle: ({gradx}, {grady}, {gradz})")

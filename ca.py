"""
Authors:     Ben de Ruijter, Rens Vester
studentId:   13192639, 12958042
Study:       BSc Informatica
Course:      Introduction Computational Science

This file contains an implementation of a cellular automaton that simulates a
forest fire.

Based on given framework.
"""

import numpy as np
import read_map
from pyics import Model
import random
import time

T_TYPES = {'BOUND': -1, 'ROAD': 0, 'VEG': 1, 'FIRE': 2, 'BURNED': 3,
           'MOOR': 4, 'WATER': 5, 'STOP': 6}
V_TYPES = {'TREE': 1, 'MOOR': 0, 'NONE': -1}
THETA = [[3/4 * np.pi, 1/2 * np.pi, 1/4 * np.pi],
         [np.pi, 0, 0],
         [5/4 * np.pi, 3/2 * np.pi, 7/4 * np.pi]]


class Vegetation():
    def __init__(self, v_type=V_TYPES['TREE']):
        self.v_type = v_type

        if self.v_type == V_TYPES['TREE']:
            self.time_to_burn = 3
            self.p_veg = 0.4
        elif self.v_type == V_TYPES['MOOR']:
            self.time_to_burn = 1
            self.p_veg = 0
        else:
            self.time_to_burn = 1
            self.p_veg = 0

        self.p_den = np.random.choice([0, 0.3])
        self.time_burning = 0


class Tile():
    def __init__(self, t_type=T_TYPES['ROAD'], veg=None):
        self.t_type = t_type

        if self.t_type == 1 and veg == V_TYPES['MOOR']:
            self.veg = Vegetation(V_TYPES['MOOR'])
        elif self.t_type == 1 and veg == V_TYPES['TREE']:
            self.veg = Vegetation(V_TYPES['TREE'])
        elif self.t_type == T_TYPES['ROAD']:
            self.veg = Vegetation(V_TYPES['NONE'])
        else:
            self.veg = None

        self.t_humidity = CASim().humidity


class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.config = None
        self.total_vegetation = 0
        self.burned = 0

        self.make_param('width', 50)
        self.make_param('height', 50)
        self.make_param('steps', 150)
        self.make_param('humidity', 0.5, setter=self.setter_double)
        self.make_param('wind_speed', 12)
        self.make_param('wind_direction', 50, setter=self.setter_wind)
        self.make_param('random_forest', False)
        self.make_param('density', 0.8)
        self.make_param('vegetation_density', 0.5)
        self.make_param('firebreak_offs', '20, 100', setter=self.setter_fb)
        self.make_param('firebreak_len', 0)

    def setter_double(self, val):
        """Setter humidity, clipping value between 0 and 1."""
        return(max(0.0, min(val, 1.0)))

    def setter_wind(self, val):
        """Setter wind, clipping value between 0 and 360. These extremes meaning from north to south. """
        return(max(0, min(val, 360)))

    def setter_fb(self, val):
        """Setter for firebreak offset, returning tuple."""
        res = val
        if val and val[0] == '(':
            res = val[1:]
        if val and val[-1] == ')':
            res = res[:-1]
        return(tuple(map(int, res.split(','))))

    def check(self, inp, cur):
        """Calculates the new state of the current Tile."""
        angle = (self.wind_direction + 180) * (np.pi / 180)
        if cur.t_type == T_TYPES['VEG'] or cur.t_type == T_TYPES['ROAD']:
            for i, row in enumerate(inp):
                for j, nb in enumerate(row):
                    if i == 1 and j == 1:
                        # The cell is the current Tile
                        break

                    # Calculate the burn probability.
                    f_t = np.exp(self.wind_speed * 0.131 *
                                 (np.cos(abs(angle - THETA[j][i])) - 1))
                    p_w = np.exp(0.045 * self.wind_speed) * f_t
                    p_burn = (1 - self.humidity) * ((1 + cur.veg.p_den) *
                                                    (1 + cur.veg.p_veg) * p_w)

                    if cur.t_type == T_TYPES['VEG'] and nb.t_type == T_TYPES['FIRE'] and np.random.random() > (1 - p_burn):
                        # Current Tile is set ablaze.
                        return T_TYPES['FIRE']
                    elif cur.t_type == T_TYPES['ROAD'] and nb.t_type == T_TYPES['FIRE'] and np.random.random() > (1 - p_burn)*2:
                        # Chance that fire spreads over a road.
                        return T_TYPES['FIRE']
        elif cur.t_type == T_TYPES['FIRE']:
            cur.veg.time_burning += 1
            if cur.veg.time_burning >= cur.veg.time_to_burn:
                # Current Tile is burned out, unless road, roads cant burn.
                if cur.veg.v_type == V_TYPES['NONE']:
                    cur.veg.time_burning = 0
                    cur.t_type = T_TYPES['ROAD']
                    return T_TYPES['ROAD']
                else:
                    return T_TYPES['BURNED']
        return cur.t_type

    def setup_de_mein(self):
        """ Satellite images of national park 'de Meinweg'. Water, roads and vegetation is loaded in from a generated bitmap downscaled by 5. Original resolution of the sat photos was 1460x815. Resulting image is 292x163. The config is returned with coordinates of the starting point of the forest fire on April 20th 2020."""

        wegenkaart = 'Atlaskaarten_Meinweg/WegenkaartPS.png'
        waterkaart = 'Atlaskaarten_Meinweg/WaterkaartPS.png'
        luchtfoto = 'Atlaskaarten_Meinweg/Luchtfoto.png'

        # Scale percentage. Full resolution would take up too much resources.
        p = 20

        water = read_map.bitmap_water_scaled(waterkaart, p)
        self.height, self.width = water.shape
        water = np.ndarray.flatten(water)
        road = np.ndarray.flatten(read_map.bitmap_road_scaled(wegenkaart, p))
        tree = np.ndarray.flatten(read_map.bitmap_trees_scaled(luchtfoto, p))

        size = self.height * self.width

        # Fill the configuration with Tiles that indicate a boundary.
        config = np.full((2 + self.height, 2 + self.width),
                         Tile(T_TYPES['BOUND']))

        # Initialize t_types with Moor landscape.
        t_types = [(1, 0)] * size
        for i in range(size):
            # Road.
            if road[i]:
                t_types[i] = (0, None)
            # Water.
            elif water[i]:
                t_types[i] = (5, None)
            # Tree.
            elif tree[i]:
                t_types[i] = (1, 1)

        config[1:-1, 1:-1] = np.array([Tile(*t_type) for t_type in t_types]
                                      ).reshape(self.height, self.width).reshape(self.height, self.width)

        return config, 230, 36

    def setup_random_forest(self):
        """Set up a random initial configuration."""
        # Fill the configuration with Tiles that indicate a boundary.
        config = np.full((2 + self.height, 2 + self.width),
                         Tile(T_TYPES['BOUND']))

        # Probabilities of choosing a clear or vegetation Tile. Based on the
        # density parameter
        t_probs = [1 - self.density, self.density * (1 - self.vegetation_density),
                   self.density * self.vegetation_density]

        indices = [0, 1, 2]
        t_types = [(0, None), (1, 1), (1, 0)]

        t_types_indices = np.random.choice(indices, size=self.height * self.width,
                                           p=t_probs)

        temp = np.array([Tile(*t_types[index]) for index in t_types_indices])
        config[1:-1, 1:-1] = temp.reshape(self.height, self.width)

        # Pick a random starting point of the fire. The fire can only be
        # started on a vegetation Tile.
        cells = np.vectorize(lambda x: x.t_type)(config[1:-1, 1:-1])
        possible_fire = list(zip(*np.where(cells == T_TYPES['VEG'])))
        y, x = random.choice(possible_fire)

        return config, x, y

    def firebreak(self, width=2):
        """Tanks from the ministry defense department create a so called firebreak. Fire should not be able to move past this line because all sources of fuel have been removed."""
        x_off, y_off = self.firebreak_offs
        length = self.firebreak_len
        for i in range(x_off, x_off+width, 1):
            for j in range(y_off - int(length/2), y_off + length, 1):
                if self.config[j][i+(j-y_off)].veg:
                    self.config[j][i+(j-y_off)].t_type = T_TYPES['STOP']
                    self.config[j][i+(j-y_off)].veg.v_type = V_TYPES['NONE']
                    self.config[j][i+(j-y_off)].veg.time_to_burn = 1

    def setup_initial_config(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first configuration."""
        if self.random_forest:
            self.config, x, y = self.setup_random_forest()
        else:
            self.config, x, y = self.setup_de_mein()
            if self.firebreak_len:
                self.firebreak()

        # Caculate the neighbourhood of the starting fire based on the starting
        # point coordinates (y, x).
        indices = [i + 1 for i in range(x - 2, x + 2)
                   if i <= self.width + 1]
        neighbourhood = [self.config[y + 1 + i, indices] for i in range(-2, 2)
                         if y + i <= self.height + 1]

        # Set up inital fire based on the starting point coordinates (y, x).
        n = 0
        for row in neighbourhood:
            for tile in row:
                n += 1
                if (n not in [1, 4, 13, 16] and
                        tile.t_type != T_TYPES['ROAD'] and
                        tile.t_type != T_TYPES['BOUND']):
                    tile.t_type = T_TYPES['FIRE']

    def calc_total_vegetation(self):
        """Returns the total amount of vegetation Tiles."""
        cell_t_types = np.vectorize(lambda x: x.t_type)(self.config)

        return np.count_nonzero(cell_t_types == T_TYPES['VEG'])

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.setup_initial_config()
        self.total_vegetation = self.calc_total_vegetation()

    def calc_burned(self):
        """Returns the percentage burned vegetation."""
        cell_t_types = np.vectorize(lambda x: x.t_type)(self.config)
        burned_count = np.count_nonzero(cell_t_types == T_TYPES['BURNED'])

        return round((burned_count / self.total_vegetation) * 100, 2)

    def draw(self):
        """Draws the current state of the grid."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()

        colors = [(0.4, 0.4, 0.4), (0.2, 0.3, 0), 'tab:orange',
                  (0.12, 0.1, 0.1), (0.4, 0.5, 0.1), (0.1, 0.3, 0.7), 'bisque']

        my_cmap = ListedColormap(colors)
        cells = np.vectorize(lambda x: x.t_type)(self.config[1:-1, 1:-1])

        for y, row in enumerate(self.config[1:-1, 1:-1]):
            for x, col in enumerate(row):
                if col.t_type == 1 and col.veg and col.veg.v_type == 1:
                    cells[y][x] = 4

        plt.imshow(cells, interpolation='none', vmin=0,
                   vmax=6, cmap=my_cmap)

        plt.axis('image')
        plt.title('t = %d' % self.t)

    def check_burning(self):
        """Returns True if the forest is still burning. False otherwise."""
        config_types = np.vectorize(lambda x: x.t_type)(self.config)
        fire = np.where(config_types == T_TYPES['FIRE'])

        return fire[0].size != 0

    def apply_changes(self, changes, changes_indices):
        for y, x in changes_indices:
            self.config[y][x].t_type = changes[y][x]

    def calc_active_fire(self):
        tile_types = np.vectorize(lambda x: x.t_type)(self.config)

        return list(zip(*np.where(tile_types == T_TYPES['FIRE'])))

    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""
        cur_time = time.time()
        self.t += 1
        if self.t >= self.steps or not self.check_burning():
            self.burned = self.calc_burned()
            return True

        active_fire = self.calc_active_fire()

        changes = np.zeros((self.height + 2, self.width + 2), dtype=np.int8)

        for y, x in active_fire:
            indices = [i + 1 for i in range(x - 2, x + 1) if i <= self.width]
            neighbourhood = [self.config[y + i, indices] for i in range(-1, 2)
                             if y + i <= self.height]
            for y2, row in enumerate(neighbourhood, -1):
                for x2, cell in enumerate(row, -1):
                    indices2 = [i + 1 for i in range(x2 + x - 2, x2 + x + 1)
                                if x2 + i <= self.width]
                    neighbourhood2 = [self.config[y + y2 + i, indices2] for i in range(-1, 2)
                                      if y2 + y + i <= self.height]
                    result = self.check(neighbourhood2, cell)
                    if result != self.config[y + y2][x + x2]:
                        changes[y + y2][x + x2] = result

        changes_indices = list(zip(*np.nonzero(changes)))

        self.apply_changes(changes, changes_indices)


if __name__ == '__main__':
    sim = CASim()
    from pyics import GUI
    cx = GUI(sim)
    cx.start()

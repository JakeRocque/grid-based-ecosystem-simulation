"""
File: Field.py
Description: A field where an ecosystem simulation takes place.
"""

import numpy as np
from scipy.ndimage import uniform_filter

import matplotlib.pyplot as plt

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from backend.config.species_config import species_config


# -------- Field --------
class Field:
    """
    Field for a life simulation to take place. 
    It maintains the states of all animals on the field according to the given species_config.
    """
    def __init__(
            self, width=200, height=150, 
            grass_rate=6, 
            starvation_energy=25,
            grass_energy=30,
            reproduce_energy=25,
            crowd_penalty=25,
            s_config=species_config,
            pop_cap=25000
        ):
        """
        Initialize an empty Field.
        
        :param self:              Field.
        :param width:             Cell width of simulation grid.
        :param height:            Cell height of simulation grid.
        :param grass_rate:        Chance that grass will regrow during a given cycle (0-10).
        :param starvation_energy: Energy level at which an animal will exhibit starvation 
                                  behavior (0-100).
                                  This includes ignoring predators for food.
        :param grass_energy:      The base energy grass will provide to animals which eat it, 
                                  prior to metabolism (0-100).
        :param reproduce_energy:  The base energy above which an animal would reproduce (0-100).
        :param crowd_penalty:     The additional energy above reproduce_energy needed to 
                                  reproduce on a given cell for every other animal on it (0-100).
        :param s_config:          The dictionary containing all attributes for animals of the same
                                  species: name, color, e (starting energy), metabolism, 
                                  hunger_rate, eat_range, sense_range, sense_change, litter, prey
        :param pop_cap            Maximum number of allowed animals on this Field at any time.
                                  If going to be exceeded, random animals will be removed.
        """
        if not isinstance(width, int) or not isinstance(height, int) or width < 25 or height < 25:
            raise ValueError("Error: Width or height is too small.")
        if not isinstance(grass_rate, int) or grass_rate < 0 or grass_rate > 10:
            raise ValueError("Error: grass_rate must be between 0 and 10.")
        if not isinstance(starvation_energy, int) or starvation_energy < 0 or starvation_energy > 100:
            raise ValueError("Error: starvation_energy must be between 0 and 100.")
        if not isinstance(grass_energy, int) or grass_energy < 0 or grass_energy > 100:
            raise ValueError("Error: grass_energy must be between 0 and 100.")
        if not isinstance(reproduce_energy, int) or reproduce_energy < 0 or reproduce_energy > 100:
            raise ValueError("Error: reproduce_energy must be between 0 and 100.")
        if not isinstance(crowd_penalty, int) or crowd_penalty < 0 or crowd_penalty > 100:
            raise ValueError("Error: crowd_penalty must be between 0 and 100.")
        if not isinstance(pop_cap, int) or pop_cap < 1:
            raise ValueError("Error: pop_cap must be greater than 0.")
        
        if not isinstance(s_config, dict):
            raise ValueError("Error: s_config must be a dictionary.")
        for sid, config in s_config.items():
            if not isinstance(sid, int) or sid < 0:
                raise ValueError("Error: species ids must be non-negative integers.")
            if (
                'name' not in config 
                or 'color' not in config 
                or 'e' not in config 
                or 'movement' not in config 
                or 'metabolism' not in config 
                or 'hunger_rate' not in config 
                or 'eat_range' not in config 
                or 'sense_range' not in config 
                or 'sense_chance' not in config 
                or 'litter' not in config 
                or 'prey' not in config):
                raise ValueError(f"Error: species config for id {sid} is missing required fields.")

        self.width = width
        self.height = height
        self.generation_num = 0

        self.rng = np.random.default_rng()

        self.grass_rate = grass_rate
        self.starvation_energy = starvation_energy
        self.grass_energy = grass_energy
        self.reproduce_energy = reproduce_energy
        self.crowd_penalty = crowd_penalty

        # Grass grid: 1 = grass, 0 = dirt
        self.field = np.ones((width, height), dtype=np.intp)

        # All animals in one array: [x, y, energy, species_id]
        # This data is unique per individual
        animal_dtype = np.dtype([
            ('x', np.int32),
            ('y', np.int32),
            ('energy', np.int32),
            ('species', np.int32)
        ])
        self.animal_data = np.empty(0, dtype=animal_dtype)

        # Species-level constants indexed by species_id
        self.s_config = s_config

        # Set hard population limit for performance reasons.
        self.pop_cap = pop_cap

        # Pre-compute predator lists for each species
        self._predator_map = {sid: [] for sid in self.s_config}
        for sid, config in self.s_config.items():
            for pid in config['prey']:
                if pid in self._predator_map:
                    self._predator_map[pid].append(sid)

        # Pre-compute eating radii
        self._eat_radii = {}
        for sid, config in self.s_config.items():
            self._eat_radii[sid] = config.get('eat_range', 0)

        # Population history per species (lists during sim, convert later)
        self.population_history = {sid: [] for sid in self.s_config}

    def add_animals(self, species_id, count):
        """
        Add a specified amount of a given animal (using its species id) to add to the Field.
        
        :param self: Field.
        :param species_id: Dictionary id of animal in s_config.
        :param count: How many of the specified animal to add.
        """
        if not isinstance(count, int) or count < 1:
            raise ValueError("Error: count must be a positive integer.")
        if species_id not in self.s_config:
            raise ValueError(f"Error: species_id {species_id} is not in s_config.")

        config = self.s_config[species_id]
        
        new_animals = np.zeros(count, dtype=self.animal_data.dtype)
        new_animals['x'] = self.rng.integers(0, self.width, size=count)
        new_animals['y'] = self.rng.integers(0, self.height, size=count)
        new_animals['energy'] = config['e']
        new_animals['species'] = species_id
        self.animal_data = np.concatenate([self.animal_data, new_animals])

    def remove_animals(self, species_id, count):
        """
        Add a specified amount of a given animal (using its species id) to remove from the Field.
        
        :param self: Field.
        :param species_id: Dictionary id of animal in s_config.
        :param count: How many of the specified animal to remove.
        """
        if not isinstance(count, int) or count < 1:
            raise ValueError("Error: count must be a positive integer.")
        if species_id not in self.s_config:
            raise ValueError(f"Error: species_id {species_id} is not in s_config.")
        
        indices = np.where(self.animal_data['species'] == species_id)[0]
        count = min(count, len(indices))
        if count == 0:
            return
        
        to_remove = self.rng.choice(indices, size=count, replace=False)
        mask = np.ones(len(self.animal_data), dtype=bool)
        mask[to_remove] = False
        self.animal_data = self.animal_data[mask]

    def _build_gradient(self, species_ids, sense_range):
        """
        Build a gradient pointing toward the specified species within the given sense range.
        
        :param self: Field.
        :param species_ids: species to include in gradient 
                            (e.g. prey for attraction, predators for repulsion, -1 for grass)
        :param sense_range: How far away animals can sense to be included in the gradient.
        :return: Tuple of (gx, gy) gradients in x and y directions.
        """
        size = 2 * sense_range + 1
        attract = np.zeros((self.width, self.height), dtype=np.float64)
        
        for sid in species_ids:
            if sid == -1:
                attract += uniform_filter(self.field.astype(np.float64), size=size, mode='wrap') * (size * size)
            else:
                prey_mask = self.animal_data['species'] == sid
                if prey_mask.sum() == 0:
                    continue
                grid = np.zeros((self.width, self.height), dtype=np.float64)
                np.add.at(grid, (self.animal_data[prey_mask]['x'], self.animal_data[prey_mask]['y']), 1)
                attract += uniform_filter(grid, size=size, mode='wrap') * (size * size)
        
        gx = np.sign(np.roll(attract, -1, axis=0) - np.roll(attract, 1, axis=0))
        gy = np.sign(np.roll(attract, -1, axis=1) - np.roll(attract, 1, axis=1))
        return gx, gy

    def move(self):
        """
        Move animals on the field according to their movement attribute 
        and the presence of food and predators.
        
        :param self: Field.
        """
        for sid, config in self.s_config.items():            
            # iterate over each species and make boolean mask
            mask = self.animal_data['species'] == sid
            count = mask.sum()
            if count == 0:
                continue

            subset = self.animal_data[mask]
            r = config.get('sense_range', 3)

            dx = np.zeros(count, dtype=int)
            dy = np.zeros(count, dtype=int)

            # Repel from predators
            predators = self._predator_map[sid]
            sense_chance = config.get('sense_chance', 70)
            senses = self.rng.random(count) < (sense_chance / 100)

            if predators and any((self.animal_data['species'] == p).sum() > 0 for p in predators):
                px, py = self._build_gradient(predators, r)
                dx[senses] = -px[subset['x'][senses], subset['y'][senses]].astype(int)
                dy[senses] = -py[subset['x'][senses], subset['y'][senses]].astype(int)

            # If senses, attract toward food only where no threat or starving
            starving = subset['energy'] <= self.starvation_energy 
            no_threat = ((dx == 0) & (dy == 0) & senses) | (starving & senses)
            if no_threat.any():
                gx, gy = self._build_gradient(config['prey'], r)
                dx[no_threat] = gx[subset['x'][no_threat], subset['y'][no_threat]].astype(int)
                dy[no_threat] = gy[subset['x'][no_threat], subset['y'][no_threat]].astype(int)

            dx = np.clip(dx, -1, 1)
            dy = np.clip(dy, -1, 1)
            subset['x'] = (subset['x'] + dx) % self.width
            subset['y'] = (subset['y'] + dy) % self.height

            # create moves array randomly from config same size as current species subset
            moves = self.rng.integers(1, config['movement'] + 1, size=count)

            max_moves = moves.max()
            all_dx = self.rng.integers(-1, 2, size=(count, max_moves))
            all_dy = self.rng.integers(-1, 2, size=(count, max_moves))

            step_mask = np.arange(max_moves)[None, :] < moves[:, None]
            total_dx = (all_dx * step_mask).sum(axis=1)
            total_dy = (all_dy * step_mask).sum(axis=1)

            subset['x'] = (subset['x'] + total_dx) % self.width
            subset['y'] = (subset['y'] + total_dy) % self.height

            self.animal_data[mask] = subset

    def _make_grid(self, x, y, weights=None, kernel_radius=0):
        """
        Make a grid counting how many animals are at each (x, y) coordinate, 
        optionally weighted by energy and smoothed by a kernel.
        
        :param self: Field.
        :param x: Array of x coordinates.
        :param y: Array of y coordinates.
        :param weights: Optional array of weights for each (x, y) pair.
        :param kernel_radius: Radius of smoothing kernel (0 = no smoothing).
        :return: 2D array of counts or weighted counts at each coordinate.
        """
        grid = np.zeros((self.width, self.height), dtype=np.float64)
        np.add.at(grid, (x, y), 1 if weights is None else weights)
        if kernel_radius > 0:
            size = 2 * kernel_radius + 1
            grid = uniform_filter(grid, size=size, mode='wrap') * (size * size)
            grid = grid.astype(int)
        return grid

    def _eat_grass(self, preds, predator_mask, config, pred_count, kernel_radius):
        """
        Handle eating grass for a set of animals.
        
        :param self: Field.
        :param preds: Array of predator animals.
        :param predator_mask: Boolean mask for predator animals.
        :param config: Configuration for the predator species.
        :param pred_count: Description
        :param kernel_radius: Description
        """
        if kernel_radius > 0:
            size = 2 * kernel_radius + 1
            grass_nearby = uniform_filter(self.field.astype(np.float64), size=size, mode='wrap') * (size * size)
        else:
            grass_nearby = self.field.astype(np.float64)
    
        shares = (grass_nearby * self.grass_energy) // np.maximum(pred_count, 1)
        energy_gain = (shares[preds['x'], preds['y']] * config['metabolism'] // 100).astype(np.int32)
        preds['energy'] = np.minimum(preds['energy'] + energy_gain, 100)
        self.animal_data[predator_mask] = preds
        self.field[(pred_count > 0) & (self.field > 0)] = 0

    def _eat_prey(self, preds, predator_mask, config, pred_count, pid, kernel_radius, alive):
        """
        Handle eating prey for a set of animals.
        
        :param self: Field.
        :param preds: Array of predator animals.
        :param predator_mask: Boolean mask for predator animals.
        :param config: Configuration for the predator species.
        :param pred_count: Count of predators at each coordinate.
        :param pid: ID of the prey species.
        :param kernel_radius: Radius of smoothing kernel for prey grid.
        :param alive: Boolean array indicating which animals are alive.
        """
        prey_mask = (self.animal_data['species'] == pid) & alive
        if prey_mask.sum() == 0:
            return
        prey = self.animal_data[prey_mask]

        prey_grid = self._make_grid(prey['x'], prey['y'], prey['energy'], kernel_radius)
        shares = prey_grid // np.maximum(pred_count, 1)
        energy_gain = (shares[preds['x'], preds['y']] * config['metabolism'] // 100).astype(np.int32)
        preds['energy'] = np.minimum(preds['energy'] + energy_gain, 100)
        self.animal_data[predator_mask] = preds

        eaten = pred_count[prey['x'], prey['y']] > 0
        alive[np.where(prey_mask)[0][eaten]] = False

    def eat(self):
        """
        For each species, handle eating behavior according to its prey list and metabolism.
        
        :param self: Field.
        """
        alive = np.ones(len(self.animal_data), dtype=bool)
        for sid, config in self.s_config.items():
            predator_mask = (self.animal_data['species'] == sid) & alive
            if predator_mask.sum() == 0:
                continue

            r = self._eat_radii[sid]
            preds = self.animal_data[predator_mask]
            pred_count = self._make_grid(preds['x'], preds['y'], kernel_radius=r)

            for pid in config['prey']:
                if pid == -1:
                    self._eat_grass(preds, predator_mask, config, pred_count, r)
                else:
                    self._eat_prey(preds, predator_mask, config, pred_count, pid, r, alive)
                    
        self.animal_data = self.animal_data[alive]

    def survive(self):
        """
        For each animal, reduce energy by its hunger_rate and remove if energy falls below 0.
        
        :param self: Field.
        """
        hunger_rates = np.array([self.s_config[sid]['hunger_rate'] for sid in self.s_config])
        self.animal_data['energy'] -= hunger_rates[self.animal_data['species']]
        self.animal_data = self.animal_data[self.animal_data['energy'] >= 0]
            

    def reproduce(self):
        """
        For each animal above reproduce_energy, create babies according to litter size,
        applying crowd_penalty based on local density.
        
        :param self: Field.
        """
        all_babies = []

        # Build a density grid: how many animals are at each (x, y)
        density = np.zeros((self.width, self.height), dtype=np.int32)
        np.add.at(density, (self.animal_data['x'], self.animal_data['y']), 1)

        for sid, config in self.s_config.items():
            # iterate over each species and make boolean mask
            mask = self.animal_data['species'] == sid
            count = mask.sum()
            if count == 0:
                continue

            subset = self.animal_data[mask]

            # Base threshold + penalty per neighbor on the same cell
            base_threshold = config.get('reproduce_threshold', self.reproduce_energy)
            crowd_penalty = config.get('crowd_penalty', self.crowd_penalty)
            local_density = density[subset['x'], subset['y']]
            thresholds = base_threshold + crowd_penalty * (local_density - 1)

            will_reproduce = subset['energy'] > thresholds

            if will_reproduce.sum() == 0:
                continue

            parents = subset[will_reproduce]
            baby_energy = self.rng.integers(1, parents['energy'] + 1)

            litter_sizes = self.rng.choice(config['litter'], size=len(parents))

            baby_x = np.repeat(parents['x'], litter_sizes)
            baby_y = np.repeat(parents['y'], litter_sizes)
            baby_energies = np.repeat(baby_energy // litter_sizes, litter_sizes)

            babies = np.zeros(len(baby_x), dtype=self.animal_data.dtype)
            babies['x'] = baby_x
            babies['y'] = baby_y
            babies['energy'] = baby_energies
            babies['species'] = sid

            all_babies.append(babies)        

            # reproduction energy loss
            subset['energy'][will_reproduce] = np.maximum(
                subset['energy'][will_reproduce] - baby_energy, 0)

            self.animal_data[mask] = subset

        if all_babies:
            self.animal_data = np.concatenate([self.animal_data] + all_babies)
        

    def grow(self):
        """
        For each cell, regrow grass with a chance based on grass_rate.
        
        :param self: Field.
        """
        self.field |= (self.rng.random((self.width, self.height)) < (self.grass_rate / 100))

    def _enforce_population_cap(self):
        """
        Randomly removes animals to maintain a hard animal number limit for performance reasons.
        
        :param self: Field.
        :param cap: Maximum number of animals allowed on this Field.
        """
        if len(self.animal_data) > self.pop_cap:
            keep = self.rng.choice(len(self.animal_data), size=self.pop_cap, replace=False)
            self.animal_data = self.animal_data[keep]

    def generation(self):
        """
        Advance the simulation by one generation: move, eat, survive, reproduce, grow grass,
        and enforce population cap. Also update generation number and population history.
        
        :param self: Field.
        """
        self.move()
        self.eat()
        self.survive()
        self.reproduce()
        self._enforce_population_cap()
        self.grow()

        self.generation_num += 1

        for sid in self.s_config:
            self.population_history[sid].append((self.animal_data['species'] == sid).sum())

    def get_array(self):
        """
        Return a numpy array representation of the field and animals.
        
        :param self: Field.
        """
        grid = self.field.copy()
        data = self.animal_data

        valid_sids = np.array(list(self.s_config.keys()))
        mask = np.isin(data['species'], valid_sids)
        subset = data[mask]

        if len(subset) == 0:
            return grid

        # values = subset['species'] + 2
        # order = np.argsort(values)

        energy = np.clip(subset['energy'], 0, 99)
        values = 2 + subset['species'] * 100 + energy
        order = np.argsort(subset['species'])

        # write in ascending order — highest value wins via last write
        grid[subset['x'][order], subset['y'][order]] = values[order]

        return grid

    def plot_animals(self):
        """
        Plot population history for each species.
        
        :param self: Field.
        """
        for sid, config in self.s_config.items():
            plt.plot(range(self.generation_num), self.population_history[sid],
                    color=config['color'], label=config['name'])
            
        max_pop = max(max(h) for h in self.population_history.values() if h)
        plt.ylim(0, int(max_pop * 1.15))
            
        plt.legend()

        ax = plt.gca()
        ax.set_facecolor('lightgray')

        plt.show()
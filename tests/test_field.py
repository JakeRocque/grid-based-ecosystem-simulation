import pytest

import numpy as np

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from backend.Field import Field


# -------- Examples --------
config_1 = {
    0: {"name": "rabbit",
        "color" : "white",
        "e": 75, "movement": 1,
        "metabolism": 65,
        "hunger_rate": 4,
        "eat_range": 0,
        "sense_range": 2,
        "sense_chance": 45,
        "litter": [2, 3, 4, 5, 6],
        "prey": [-1]},
    1: {"name": "fox",
        "color": "red",
        "e": 50,
        "movement": 3,
        "metabolism": 90,
        "hunger_rate": 6,
        "eat_range": 2,
        "sense_range": 4,
        "sense_chance": 90,
        "litter": [1, 2],
        "prey": [0]},
    2: {"name": "wolf",
        "color": "blue",
        "e": 50,
        "movement": 2,
        "metabolism": 80,
        "hunger_rate": 13,
        "eat_range": 3,
        "sense_range": 3,
        "sense_chance": 75,
        "litter": [2, 3],
        "prey": [0, 1]},
}


# -------- Tests --------
def test_constructor_invalid_types(): 
    with pytest.raises(ValueError):
        Field(width="invalid")
    with pytest.raises(ValueError):
        Field(height="invalid")
    with pytest.raises(ValueError):
        Field(grass_rate="invalid")
    with pytest.raises(ValueError):
        Field(starvation_energy="invalid")
    with pytest.raises(ValueError):
        Field(grass_energy="invalid")
    with pytest.raises(ValueError):
        Field(reproduce_energy="invalid")
    with pytest.raises(ValueError):
        Field(crowd_penalty="invalid")
    with pytest.raises(ValueError):
        Field(s_config="invalid")
    with pytest.raises(ValueError):
        Field(pop_cap="invalid")

    with pytest.raises(ValueError):
        Field(s_config={"invalid": "not a dict of species configs"})
    with pytest.raises(ValueError):
        Field(s_config={0: {"name": "rabbit"}})  # missing required keys in species config
    with pytest.raises(ValueError):
        Field(s_config={0: {
            "name": "rabbit", "color": "white", "e": 75, "movement": 1, 
            "metabolism": 65, "hunger_rate": 4, "eat_range": 0, 
            "sense_range": 2, "sense_chance": 45, "litter": [2, 3, 4, 5, 6]}})  # missing 'prey'

def test_constructor_invalid_inputs(): 
    with pytest.raises(ValueError):
        Field(width=24)
    with pytest.raises(ValueError):
        Field(height=24)
    with pytest.raises(ValueError):
        Field(grass_rate=-1)
    with pytest.raises(ValueError):
        Field(grass_rate=11)
    with pytest.raises(ValueError):
        Field(starvation_energy=-1)
    with pytest.raises(ValueError):
        Field(starvation_energy=101)
    with pytest.raises(ValueError):
        Field(grass_energy=-1)
    with pytest.raises(ValueError):
        Field(grass_energy=101)
    with pytest.raises(ValueError):
        Field(reproduce_energy=-1)
    with pytest.raises(ValueError):
        Field(reproduce_energy=101)
    with pytest.raises(ValueError):
        Field(crowd_penalty=-1)
    with pytest.raises(ValueError):
        Field(crowd_penalty=101)
    with pytest.raises(ValueError):
        Field(pop_cap=0)

def test_constructor_valid_inputs():
    try:
        Field(width=100, height=100, grass_rate=5, starvation_energy=50, grass_energy=20, 
              reproduce_energy=80, crowd_penalty=10, s_config={}, pop_cap=1000)
    except ValueError:
        pytest.fail("Constructor raised ValueError unexpectedly with valid inputs.")
    
    try:
        Field(width=50, height=50)
    except ValueError:
        pytest.fail("Constructor raised ValueError unexpectedly with default parameters.")

    try:
        Field(width=200, height=150, grass_rate=0, starvation_energy=0, grass_energy=0, 
              reproduce_energy=0, crowd_penalty=0, s_config={}, pop_cap=1)
    except ValueError:
        pytest.fail("Constructor raised ValueError unexpectedly with edge case inputs.")

    try:
        Field(width=200, height=150, grass_rate=10, starvation_energy=100, grass_energy=100, 
              reproduce_energy=100, crowd_penalty=100, s_config={}, pop_cap=10000)
    except ValueError:
        pytest.fail("Constructor raised ValueError unexpectedly with edge case inputs.")

def test_default_parameters():
    try:
        f = Field(width=100, height=100)
        assert f.grass_rate == 6
        assert f.starvation_energy == 25
        assert f.grass_energy == 30
        assert f.reproduce_energy == 25
        assert f.crowd_penalty == 25
        assert f.pop_cap == 25000
    except ValueError:
        pytest.fail("Constructor raised ValueError unexpectedly with default parameters.") 

def test_add_animals_invalid_inputs():
    f = Field(width=100, height=100)
    with pytest.raises(ValueError):
        f.add_animals(species_id="invalid", count=10)
    with pytest.raises(ValueError):
        f.add_animals(species_id=0, count="invalid")
    with pytest.raises(ValueError):
        f.add_animals(species_id=-1, count=10)
    with pytest.raises(ValueError):
        f.add_animals(species_id=999, count=10)
    with pytest.raises(ValueError):
        f.add_animals(species_id=0, count=0) 

def test_add_animals_valid_inputs():
    f = Field(width=100, height=100, s_config=config_1)
    f.add_animals(species_id=0, count=10)
    f.add_animals(species_id=1, count=5)
    f.add_animals(species_id=2, count=1)

    assert np.sum(f.animal_data['species'] == 0) == 10
    assert np.sum(f.animal_data['species'] == 1) == 5
    assert np.sum(f.animal_data['species'] == 2) == 1

def test_remove_animals_invalid_inputs():
    f = Field(width=100, height=100)
    with pytest.raises(ValueError):
        f.remove_animals(species_id="invalid", count=10)
    with pytest.raises(ValueError):
        f.remove_animals(species_id=0, count="invalid")
    with pytest.raises(ValueError):
        f.remove_animals(species_id=-1, count=10)
    with pytest.raises(ValueError):
        f.remove_animals(species_id=999, count=10)
    with pytest.raises(ValueError):
        f.remove_animals(species_id=0, count=0)

def test_remove_animals_valid_inputs():
    f = Field(width=100, height=100, s_config=config_1)
    f.add_animals(species_id=0, count=10)
    f.remove_animals(species_id=0, count=5)
    f.remove_animals(species_id=0, count=5)
    f.remove_animals(species_id=0, count=1)  # should not raise error even if no animals left

    assert np.sum(f.animal_data['species'] == 0) == 0
    assert np.sum(f.animal_data['species'] == 1) == 0
    assert np.sum(f.animal_data['species'] == 2) == 0

def test__build_gradient():
    f = Field(width=100, height=100, s_config=config_1)

    assert np.all(f._build_gradient(species_ids=[0], sense_range=1) == np.zeros((100, 100)))
    assert np.all(f._build_gradient(species_ids=[0], sense_range=2) == np.zeros((100, 100)))
    assert np.all(f._build_gradient(species_ids=[1], sense_range=3) == np.zeros((100, 100)))
    assert np.all(f._build_gradient(species_ids=[0, 1], sense_range=2) == np.zeros((100, 100)))
    assert np.all(f._build_gradient(species_ids=[0, 1], sense_range=3) == np.zeros((100, 100)))

def test_move():
    f = Field(width=100, height=100, s_config=config_1)
    f.add_animals(species_id=0, count=10)
    f.add_animals(species_id=1, count=5)
    f.add_animals(species_id=2, count=1)

    try:
        f.move()
    except Exception as e:
        pytest.fail(f"move() raised an unexpected exception: {e}")

def test__make_grid():
    f = Field(width=100, height=100, s_config=config_1)
    f.add_animals(species_id=0, count=10)
    f.add_animals(species_id=1, count=5)
    f.add_animals(species_id=2, count=1)

    grid = f._make_grid(10, 10)
    assert grid.shape == (100, 100)
    assert np.sum(grid == 0) == np.int64(9999)
    assert np.sum(grid == 1) == np.int64(1)
    assert np.sum(grid == 2) == np.int64(0)

def test_eat():
    f = Field(width=100, height=100, s_config=config_1)
    f.add_animals(species_id=0, count=10)
    f.add_animals(species_id=1, count=5)

    try:
        f.eat()
    except Exception as e:
        pytest.fail(f"eat() raised an unexpected exception: {e}")

def test_survive():
    f = Field(width=100, height=100, s_config=config_1)
    f.add_animals(species_id=0, count=10)
    f.add_animals(species_id=1, count=5)

    try:
        f.survive()
    except Exception as e:
        pytest.fail(f"survive() raised an unexpected exception: {e}")

def test_reproduce():
    f = Field(width=100, height=100, s_config=config_1)
    f.add_animals(species_id=0, count=10)

    try:        
        f.reproduce()
    except Exception as e:
        pytest.fail(f"reproduce() raised an unexpected exception: {e}")

def test_grow():
    f = Field(width=100, height=100, s_config=config_1)

    try:        
        f.grow()
    except Exception as e:
        pytest.fail(f"grow() raised an unexpected exception: {e}")

def test__enforce_population_cap():
    f = Field(width=100, height=100, s_config=config_1, pop_cap=10)
    f.add_animals(species_id=0, count=20)

    try:        
        f._enforce_population_cap()
    except Exception as e:
        pytest.fail(f"_enforce_population_cap() raised an unexpected exception: {e}")

    assert np.sum(f.animal_data['species'] == 0) == 10

def test_generation():
    f = Field(width=100, height=100, s_config=config_1)
    f.add_animals(species_id=0, count=10)
    f.add_animals(species_id=1, count=5)

    try:        
        f.generation()
    except Exception as e:
        pytest.fail(f"generation() raised an unexpected exception: {e}")

def test_get_array():
    f = Field(width=100, height=100, s_config=config_1)
    f.add_animals(species_id=0, count=10)
    f.add_animals(species_id=1, count=5)
    f.add_animals(species_id=2, count=1)

    try:        
        arr = f.get_array()
        assert arr.shape == (100, 100)
        assert np.sum(arr == 0) == np.int64(0)
        assert np.sum(arr == 1) == np.int64(9984)
        assert np.sum(arr == 2) == np.int64(0)
    except Exception as e:
        pytest.fail(f"get_array() raised an unexpected exception: {e}")
# Artificial Life Simulation

An interactive predator-prey ecosystem simulation with rabbits, foxes, and wolves on a 2D grid. Animals move, eat, reproduce, and die based on configurable species traits and field parameters.

## Project Structure

```
├── backend/
│   ├── Field.py                  # Core simulation logic
│   └── config/
│       └── species_config.py     # Species definitions
├── frontend/
│   └── index.html                # Web UI (WebSocket-based)
├── server.py                     # WebSocket backend
├── alife.py                      # Pyglet desktop viewer
├── requirements.txt
├── render.yaml                   # Render deployment config
└── tests/
    └── test_field.py
```

### Controls

| Key / Button | Action |
|---|---|
| `Space` | Pause / Resume |
| `+` / `-` | Speed up / slow down |
| `R` | Reset simulation |
| `Esc` | Quit (desktop only) |

---

## Species Config

Defined in `backend/config/species_config.py`. Each species is keyed by an integer ID:

```python
species_config = {
    0: {"name": "rabbit",  "color": "white", "e": 75,  "movement": 1, "metabolism": 65,
        "hunger_rate": 4,  "eat_range": 0, "sense_range": 2, "sense_chance": 45,
        "litter": [2,3,4,5,6], "prey": [-1]},          # -1 = grass
    1: {"name": "fox",     "color": "red",   "e": 50,  "movement": 3, "metabolism": 90,
        "hunger_rate": 6,  "eat_range": 2, "sense_range": 4, "sense_chance": 90,
        "litter": [1,2],   "prey": [0]},
    2: {"name": "wolf",    "color": "blue",  "e": 50,  "movement": 2, "metabolism": 80,
        "hunger_rate": 13, "eat_range": 3, "sense_range": 3, "sense_chance": 75,
        "litter": [2,3],   "prey": [0,1]},
}
```

### Species Fields

| Field | Description |
|---|---|
| `name` | Display name |
| `color` | Matplotlib color string |
| `e` | Starting energy (0–100) |
| `movement` | Max random steps per generation |
| `metabolism` | % of food energy absorbed |
| `hunger_rate` | Energy lost per generation |
| `eat_range` | Radius for eating (0 = same cell only) |
| `sense_range` | Radius for detecting food/predators |
| `sense_chance` | % chance of sensing each generation |
| `litter` | List of possible offspring counts (sampled randomly) |
| `prey` | List of prey species IDs (`-1` = grass) |

## Field Parameters

Configurable via `Field(...)` or the web UI's Field Config panel:

| Parameter | Default | Range | Description |
|---|---|---|---|
| `grass_rate` | 6 | 0–10 | Grass regrowth chance per cell per generation |
| `starvation_energy` | 25 | 0–100 | Energy below which animals ignore predators to seek food |
| `grass_energy` | 30 | 0–100 | Base energy grass provides |
| `reproduce_energy` | 25 | 0–100 | Minimum energy required to reproduce |
| `crowd_penalty` | 25 | 0–100 | Extra energy cost per neighbor when reproducing |
| `pop_cap` | 25000 | ≥1 | Hard population limit (enforced randomly) |

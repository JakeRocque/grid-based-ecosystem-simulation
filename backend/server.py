"""
File: server.py
Description: Server to send life simulation data to the frontend.
"""

import asyncio
import struct
import numpy as np
import matplotlib.colors as mcolors
import websockets
from urllib.parse import parse_qs

import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from backend.Field import Field
from backend.config.species_config import species_config


# -------- Constants --------
WIDTH, HEIGHT = 300, 200

# Defaults matching Field.__init__
FIELD_DEFAULTS = {
    'grass_rate':         6,
    'starvation_energy':  25,
    'grass_energy':       30,
    'reproduce_energy':   25,
    'crowd_penalty':      25,
}

# Color LUT
N_SHADES = 100
base_colors = ['#3b2507', 'green']
lut = np.zeros((2 + len(species_config) * N_SHADES, 3), dtype=np.uint8)
lut[0] = tuple(int(v * 255) for v in mcolors.to_rgb(base_colors[0]))
lut[1] = tuple(int(v * 255) for v in mcolors.to_rgb(base_colors[1]))
for sid in sorted(species_config):
    base_rgb = np.array(mcolors.to_rgb(species_config[sid]['color']))
    dark, bright = base_rgb * 0.75, base_rgb
    for e in range(N_SHADES):
        t = e / (N_SHADES - 1)
        lut[2 + sid * N_SHADES + e] = ((dark + t * (bright - dark)) * 255).astype(np.uint8)


# -------- Server --------
def parse_field_config(qs_string: str) -> dict:
    """Parse a query-string like 'grass_rate=6&starvation_energy=25' into
    a kwargs dict suitable for Field(), validated against FIELD_DEFAULTS."""
    cfg = dict(FIELD_DEFAULTS)  # start from defaults
    if qs_string:
        parsed = parse_qs(qs_string)
        for key, default in FIELD_DEFAULTS.items():
            if key in parsed:
                try:
                    val = float(parsed[key][0])
                    # Coerce to int for integer defaults
                    cfg[key] = int(val) if isinstance(default, int) else val
                except ValueError:
                    pass  # keep default on bad input
    return cfg


def create_field(cfg: dict | None = None):
    """
    Initialize field with 400 rabbits, 250 foxes, 150 wolves.
    Optional cfg dict overrides Field defaults.
    """
    kw = cfg if cfg is not None else FIELD_DEFAULTS
    f = Field(width=WIDTH, height=HEIGHT, **kw)
    f.add_animals(0, 400)
    f.add_animals(1, 250)
    f.add_animals(2, 150)
    return f


async def sim_handler(ws):
    """
    Send binary representation of field and maintain 30 FPS.

    :param ws: Websocket connection.
    """
    field = create_field()
    state = {'speed': 1, 'paused': False, 'gen': 0}

    async def send_frame():
        arr = field.get_array()
        rgb = lut[arr]  # (W, H, 3) uint8

        counts = {}
        for sid in sorted(species_config):
            counts[species_config[sid]['name']] = int(
                (field.animal_data['species'] == sid).sum()
            )

        # Header: gen (u32) + speed (u8) + paused (u8) + counts (species order, u16 each)
        header = struct.pack('<IBB',
            state['gen'],
            state['speed'],
            1 if state['paused'] else 0,
        )
        for sid in sorted(species_config):
            header += struct.pack('<H', counts[species_config[sid]['name']])

        await ws.send(header + rgb.tobytes())

    await send_frame()

    async def run_loop():
        while True:
            if not state['paused']:
                for _ in range(state['speed']):
                    field.generation()
                state['gen'] += state['speed']
                await send_frame()
            await asyncio.sleep(1 / 30)

    loop_task = asyncio.create_task(run_loop())

    try:
        async for msg in ws:
            cmd = msg.strip()
            if cmd == 'pause':
                state['paused'] = not state['paused']
                await send_frame()
            elif cmd == 'faster':
                state['speed'] = min(state['speed'] + 1, 10)
            elif cmd == 'slower':
                state['speed'] = max(state['speed'] - 1, 1)
            elif cmd.startswith('reset'):
                # Support both plain 'reset' and 'reset?key=val&...'
                qs = cmd[6:] if '?' in cmd else ''
                cfg = parse_field_config(qs)
                field = create_field(cfg)
                state['gen'] = 0
                state['paused'] = False
                await send_frame()
            elif cmd.startswith('spawn:'):
                parts = cmd.split(':')
                sid, count = int(parts[1]), int(parts[2])
                if sid in species_config and 1 <= count <= 1000:
                    field.add_animals(sid, count)
                    await send_frame()
            elif cmd.startswith('remove:'):
                parts = cmd.split(':')
                sid, count = int(parts[1]), int(parts[2])
                if sid in species_config and count >= 1:
                    field.remove_animals(sid, count)
                    await send_frame()
    finally:
        loop_task.cancel()


async def main():
    """
    Host server.
    """
    print("WebSocket server on ws://localhost:8765")
    async with websockets.serve(sim_handler, "localhost", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
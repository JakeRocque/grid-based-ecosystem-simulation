"""
File: alife.py
Description: Animation of a life simulation using pyglet
"""

import argparse
import numpy as np

import matplotlib.colors as mcolors
import pyglet
from pyglet.gl import glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST, glClearColor
from pyglet.window import key

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from backend.Field import Field
from backend.config.species_config import species_config


# -------- Constants --------
WIDTH = 300
HEIGHT = 200

# Build color lookup table
base_colors = ["#472a0f", 'green']
n_shades = 100  # one per energy level

lut = np.zeros((2 + len(species_config) * n_shades, 3), dtype=np.uint8)
lut[0] = tuple(int(v * 255) for v in mcolors.to_rgb(base_colors[0]))
lut[1] = tuple(int(v * 255) for v in mcolors.to_rgb(base_colors[1]))

for sid in sorted(species_config):
    base_rgb = np.array(mcolors.to_rgb(species_config[sid]['color']))
    dark = base_rgb * 0.8   # low energy = dark
    bright = base_rgb        # full energy = bright
    for e in range(n_shades):
        t = e / (n_shades - 1)
        color = dark + t * (bright - dark)
        idx = 2 + sid * n_shades + e
        lut[idx] = (color * 255).astype(np.uint8)


# -------- Define Functions --------
def create_field():
    f = Field(width=WIDTH, height=HEIGHT)
    f.add_animals(0, 400)
    f.add_animals(1, 250)
    f.add_animals(2, 150)
    return f


# -------- Main --------
def main():

    field = create_field()

    window = pyglet.window.Window(
        width=1280, height=720,
        resizable=True,
        caption="Generation: 0"
    )
    glClearColor(0, 0, 0, 1)

    # State
    state = {
        'speed': 1,
        'paused': False,
        'gen': 0,
    }

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.SPACE:
            state['paused'] = not state['paused']
        elif symbol in (key.EQUAL, key.PLUS, key.NUM_ADD):
            state['speed'] = min(state['speed'] + 1, 10)
        elif symbol in (key.MINUS, key.NUM_SUBTRACT):
            state['speed'] = max(state['speed'] - 1, 1)
        elif symbol == key.R:
            nonlocal field
            field = create_field()
            state['gen'] = 0
            state['paused'] = False
        elif symbol == key.ESCAPE:
            window.close()

    @window.event
    def on_draw():
        window.clear()

        arr = field.get_array()
        rgb = lut[arr[:, ::-1]].transpose(1, 0, 2)
        raw = rgb.tobytes()
        image = pyglet.image.ImageData(WIDTH, HEIGHT, 'RGB', raw)
        texture = image.get_texture()
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        # Scale to fill window while preserving aspect ratio
        win_w, win_h = window.width, window.height
        scale = min(win_w / WIDTH, win_h / HEIGHT)
        draw_w = win_w // 1.5
        draw_h = win_h // 1.5
        x_off = (win_w - draw_w) // 2
        y_off = (win_h - draw_h) // 2

        texture.blit(x_off, y_off, width=draw_w, height=draw_h)

        # HUD - species counts
        label_y = win_h - 120
        for sid in sorted(species_config):
            count = (field.animal_data['species'] == sid).sum()
            name = species_config[sid]['name']
            pyglet.text.Label(
                f"{name}: {count}",
                x=10, y=label_y,
                font_size=18,
                color=(255, 255, 255, 220)
            ).draw()
            label_y -= 40

        # HUD - generation and speed (top right)
        pyglet.text.Label(
            f"Gen: {state['gen']}",
            x=10, y=win_h - 40,
            font_size=24,
            color=(255, 255, 255, 220)
        ).draw()
        pyglet.text.Label(
            f"Speed: {state['speed']}x",
            x=10, y=win_h - 70,
            font_size=18,
            color=(255, 255, 255, 220)
        ).draw()

        # Paused indicator
        if state['paused']:
            pyglet.text.Label(
                "PAUSED",
                x=win_w // 2, y=win_h - 40,
                font_size=28,
                anchor_x='center',
                color=(255, 80, 80, 240)
            ).draw()

        # Controls hint
        pyglet.text.Label(
            "Space: pause  +/-: speed  R: reset  Esc: quit",
            x=win_w // 2, y=12,
            font_size=12,
            anchor_x='center',
            color=(255, 255, 255, 120)
        ).draw()

    def update(dt):
        if state['paused']:
            return
        for _ in range(state['speed']):
            field.generation()
        state['gen'] += state['speed']
        window.set_caption(f"Generation: {state['gen']}")

    pyglet.clock.schedule_interval(update, 1 / 25)
    pyglet.app.run()

if __name__ == '__main__':
    main()
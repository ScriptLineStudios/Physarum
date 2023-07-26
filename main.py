import pygame
import pygame_shaders
import numpy as np
import math
import random

pygame.init()
width = 800
height = 800

screen = pygame.display.set_mode((800, 800), pygame.DOUBLEBUF | pygame.OPENGL)

clock = pygame.time.Clock()

surface = pygame.Surface((width, height))
surface_shader = pygame_shaders.Shader(pygame_shaders.DEFAULT_VERTEX_SHADER, pygame_shaders.DEFAULT_FRAGMENT_SHADER, surface)

simulation_compute_shader = pygame_shaders.ComputeShader("sim.comp")
render_shader = pygame_shaders.ComputeShader("render.comp")

texture = pygame_shaders.Texture(surface, render_shader.ctx)
texture.texture.bind_to_image(0)

AGENT_NUM = 65500

def gen_initial_data():
    for i in range(AGENT_NUM):
        yield width / 2
        yield height / 2
        yield random.uniform(0, math.pi * 2)
        yield 1.0

agent_start_data = np.fromiter(gen_initial_data(), dtype="f4")
agent_buffer = simulation_compute_shader.ctx.buffer(agent_start_data)
agent_buffer.bind_to_storage_buffer(1)

trail_buffer = simulation_compute_shader.ctx.buffer(reserve = width * height * 4)
trail_buffer.bind_to_storage_buffer(2)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise SystemExit

    simulation_compute_shader.dispatch(AGENT_NUM, 1, 1)
    simulation_compute_shader.ctx.memory_barrier()
    render_shader.dispatch(width, height, 1)
    render_shader.ctx.memory_barrier()

    surface_shader.set_target_texture(texture)
    surface_shader.render_direct(pygame.Rect(0, 0, 800, 800), False)

    clock.tick(120)
    pygame.display.flip()
    pygame.display.set_caption(f"Simulation: {int(clock.get_fps())}")
import numpy as np
import h5py
import os
import sys
from phi.torch import flow
from tqdm import trange
from typing import Tuple
from pathlib import Path
from scipy.interpolate import interp1d


np.random.seed(777)
# change the simulation time to have a shorter simulation
simulation_time = 20.0
dt = 0.05
arena_size = (80, 60)
inflow_pos = (4, 30)
inflow_radius = 1
inflow_scaler = 0.2
velocity_grid_size = 0.5
smoke_grid_size = 0.25
simulation_steps = int(simulation_time / dt)



def converging_brownian_step(
    value_curr: np.ndarray,
    center: np.ndarray,
    gaussian_scale: float = 1.0,
    convergence: float = 0.5,
) -> np.ndarray:
    """Step to simulate Brownian noise with convergence towards a center.

    Parameters
    ----------
    value_curr : np.ndarray
        Current value of variables (i.e., noise) in Brownian motion.
    center : np.ndarray
        Center towards which the Brownian motion converges.
    gaussian_scale : float, optional
        Standard deviation of Gaussian noise to be added to the current
        value, by default 1.0
    convergence : float, optional
        Factor of convergence towards the center, by default 0.5.

    Returns
    -------
    np.ndarray
        Next value of variables (i.e., noise) in Brownian motion.
    """
    gaussian_center = (center - value_curr) * convergence
    value_diff = np.random.normal(
        loc=gaussian_center, scale=gaussian_scale, size=value_curr.shape
    )
    value_next = value_curr + value_diff
    return value_next


# Simulate Brownian noise and store the wind for every time step
curr_wind = np.zeros((2,))
wind_hist = [curr_wind.copy()]
for i in range(simulation_steps):
    curr_wind = converging_brownian_step(curr_wind, (0, 0), (1.2, 1.2), 1.0)
    wind_hist.append(curr_wind.copy())

# Define simulation grids
# constant velocity vector in every points
velocity = flow.StaggeredGrid(
    values=(10.0, 0.0),  # constant velocity field to the right
    extrapolation=flow.extrapolation.BOUNDARY,
    x=int(arena_size[0] / velocity_grid_size),
    y=int(arena_size[1] / velocity_grid_size),
    bounds=flow.Box(x=arena_size[0], y=arena_size[1]),
)

# choose extrapolation mode from
# ('undefined', 'zeros', 'boundary', 'periodic', 'symmetric', 'reflect')
# Zero smoke field at the beginning of the simulation
smoke = flow.CenteredGrid(
    values=0.0,
    extrapolation=flow.extrapolation.BOUNDARY,
    x=int(arena_size[0] / smoke_grid_size),
    y=int(arena_size[1] / smoke_grid_size),
    bounds=flow.Box(x=arena_size[0], y=arena_size[1]),
)

# Define inflow
inflow = inflow_scaler * flow.field.resample(
    flow.Sphere(x=inflow_pos[0], y=inflow_pos[1], radius=inflow_radius),
    to=smoke,
    soft=True,
)

def step(
    velocity_prev: flow.Grid,
    smoke_prev: flow.Grid,
    noise: np.ndarray,
    noise_magnitude: tuple[float, float] = (0.1, 2),
    dt: float = 1.0,
    inflow: flow.Grid = None,
    ) -> tuple[flow.Grid, flow.Grid]:
    """Simulate fluid dynamics by one time step.

    Parameters
    ----------
    velocity_prev : flow.Grid
        Velocity field at previous time step.
    smoke_prev : flow.Grid
        Smoke density at previous time step.
    noise : np.ndarray
        Brownian noise to be applied as external force.
    noise_magnitude : tuple[float, float], optional
        Magnitude of noise to be applied as external force in x and y
        directions, by default (0.1, 2)
    dt : float, optional
        Simulation time step, by default 1.0

    Returns
    -------
    tuple[flow.Grid, flow.Grid]
        Velocity field and smoke density at next time step.
    """
    smoke_next = flow.advect.mac_cormack(smoke_prev, velocity_prev, dt=dt) + inflow
    external_force = smoke_next * noise * noise_magnitude @ velocity_prev
    velocity_tentative = (
        flow.advect.semi_lagrangian(velocity_prev, velocity_prev, dt=dt)
        + external_force
    )
    velocity_next, pressure = flow.fluid.make_incompressible(velocity_tentative)
    return velocity_next, smoke_next

# Run fluid dynamics simulation
smoke_hist = []
for i in trange(simulation_steps):
    velocity, smoke = step(
        velocity,
        smoke,
        wind_hist[i],
        dt=dt,
        inflow=inflow,
        noise_magnitude=(0.5, 100.0),
    )
    smoke_vals = smoke.values.numpy("y,x")
    smoke_hist.append(smoke_vals)



#Interpolate simulated plume to save time
sim_timepoints = np.arange(0, simulation_time, step=dt)
smoke_hist_interp_fun = interp1d(sim_timepoints, smoke_hist, axis=0)

new_timepoints = np.linspace(0, simulation_time - dt, num=10000)
smoke_hist_interp = smoke_hist_interp_fun(new_timepoints)



# Save simulated plume

# Create an output directory relative to the current script location
output_dir = Path(__file__).parent / "outputs" / "plume_tracking" / "plume_dataset"

# Create the directories if they don't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Output will be saved to: {output_dir}")

with h5py.File(output_dir / "plume.hdf5", "w") as f:
    f["plume"] = np.stack(smoke_hist_interp).astype(np.float16)
    f["inflow_pos"] = inflow_pos
    f["inflow_radius"] = [inflow_radius]  # save as array with a single value
    f["inflow_scaler"] = [inflow_scaler]  # "
    
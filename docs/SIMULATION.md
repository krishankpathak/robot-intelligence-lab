# Sensor Simulation & Telemetry Analytics Guide

## Overview
The Interactive Sensor Lab in **ROBOT INTELLIGENCE LAB** provides real-time telemetry streaming and model inference for demonstration and educational purposes. It is an analytics emulator, not a full 3D multibody dynamics simulator.

## Dual Operating Modes

### Mode A: Real Kaggle Telemetry Replay
- Replays actual recordings from the Kaggle dataset sequence by sequence.
- Users choose a test run (e.g., `20240325_155003`) and motor identifier (`motor_1` through `motor_6`).
- Signals reflect genuine encoder readings, coil temperatures, and voltages.
- Evaluates the model against real-world mechanical degradation events recorded on the physical robot.

### Mode B: Synthetic Scenario Injection
- Generates artificial sensor readings modulated according to documented physical scenarios:
  1. **Normal Steady-State**: Sinusoidal position cycles, baseline temperature (~32°C), nominal voltage (~7220 mV).
  2. **Elevated Temperature (Bearing Overheat)**: Simulates thermal accumulation (+1.2°C/s drift) from unlubricated friction.
  3. **Voltage Sag (Heavy Load)**: Injects ~300 mV drop and expanded electrical ripple noise.
  4. **Mechanical Jam / Stall**: Locks position displacement at a high offset with rapid Joule heating.
  5. **Sensor Glitch Spikes**: Injects transient negative spikes (-32k) to simulate EMI serial bus dropouts.
  6. **Custom User Sliders**: Dynamic manual sliders to probe model response.

## Preventing Scenario Leakage
To maintain strict ML integrity:
- The scenario label (e.g. `is_fault_intended = True`) is **never** provided to the classifier pipeline.
- The model receives only the 3 physical sensor features:
  $$\mathbf{x} = [\text{position}, \text{temperature}, \text{voltage}]$$
- The dashboard displays both:
  1. What the user configured (Scenario Intention).
  2. What the model independently predicted.
- If the model disagrees with the configured scenario, the dashboard displays the disagreement honestly rather than forcing the expected answer.

## Performance & Memory Management
- A circular `collections.deque(maxlen=150)` bounds memory usage.
- As new points arrive, older points roll out smoothly.
- Time progression can be paused, resumed, or reset to 0.0s.

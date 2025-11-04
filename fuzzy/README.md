# Best Air Conditioner Controller Ever Made

## Authors

- Stefan Karczewski (s27459)
- Łukasz Ogorzałek (s27447)

## Running

Make sure you have uv and python, see "Environment Setup" in the main [README.md](../README.md) for more detailed instructions.

When inside the `fuzzy` directory:
- Run the program: `uv run control.py`

## Inputs and Outputs

AC with 3 inputs:
- temperature (current)
  - cold
  - cool
  - comfortable
  - warm
  - hot
- humidity (current)
  - dry
  - comfortable
  - sticky
- target_temperature
  - low
  - medium
  - high
2 Outputs:
- mode (0-2)
  - \[0, 1\) -> adjust_temperature
  - \[1, 2\] -> dehumidify
- fan_speed (0-100)

## TODO

- [ ] Simulation of a system using the AC controller
- [ ] Terminal visualization of the simulation

# Best Air Conditioner Controller Ever Made

## Authors

- Stefan Karczewski (s27459)
- Łukasz Ogorzałek (s27447)

## Running

Make sure you have uv and python, see "Environment Setup" in the main [README.md](../README.md) for more detailed instructions.

When inside the `fuzzy` directory:
- Run the control with raw data: `uv run control.py`
- Run the simulator: `uv run simulator.py`
  - There are default initial parameters when you just press "enter" for all prompts without entering a value
  - By default the simulator should start by dehumidifying the air and then switch to adjusting the temperature

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

# Computer Vision

## Authors

- Stefan Karczewski (s27459)
- Łukasz Ogorzałek (s27447)

## Running

Make sure you have uv and python, see "Environment Setup" in the main [README.md](../README.md) for more detailed instructions.

When inside the `6-computer-vision` directory:

- Run the program: `uv run main.py`

## Observations

- Detecting faces and eyes in a live camera feed is pretty fast, so it can be done in real-time
- Playing a video in the same main loop as processing the camera feed is possible, but it reduces the frame rate of the played video down to how fast we can process camera frames

## Example Usage

[example.mp4](https://github.com/user-attachments/assets/053b2c10-34ae-464a-8df2-afc30e4b68b7)

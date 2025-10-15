# Best Game Ever Made

## Running

Make sure you have uv and python, see "Environment Setup" in the main [README.md](../README.md) for more detailed instructions.

When inside the `game` directory:
- Run the game: `uv run main.py`
- Run tests: `uv run pytest`

## Rules

TODO)) Rules and game mechanics

## Example Gameplay

Example turn:
![Example Turn](./assets/example-turn.png)

On Player 1's turn:
- Player 1 placed their `[w]` token in column 6, which fell down to the bottom of the board
- Nothing else happened because there were no tokens to capture or score

On Player 2's turn:
- Player 2 placed their `[b]` token in column 6, which fell down to the 2nd row
- The token connected with two other `[b]` tokens (underlined in green)
- These three tokens were counted towards Player 2's score and removed from the board
- Other tokens above fell down to fill the gap

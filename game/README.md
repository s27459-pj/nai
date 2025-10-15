# Best Game Ever Made

## Running

Make sure you have uv and python, see "Environment Setup" in the main [README.md](../README.md) for more detailed instructions.

When inside the `game` directory:
- Run the game: `uv run main.py`
- Run tests: `uv run pytest`

## Rules

- The game board is a 7 row by 5 column grid
- Each board cell can have one token at a time or be empty
- The board has gravity - tokens fall down to fill in empty spaces below them

1. Players can place tokens in any non-full column during their turn
2. Tokens fall down to the first available row in the column
3. If a token lands on exactly two tokens of the same color, they are captured[^1] by the Player that placed the token on top
4. If there are 3+ tokens of the same color in a column or row, they are removed from the board
5. For each cleared token, the Player receives a point
6. The game ends when all possible spaces are used or Players run out of moves
7. The Player with the highest score at the end of the game wins

[^1]: Capture - Two opponent tokens below the token that landed on them are converted to be tokens of the Player that placed the token on top

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

from __future__ import annotations

import time
from typing import Callable

from easyAI.AI.Negamax import Negamax
from easyAI.Player import AI_Player

from game import BestGameEverMade


def run_game(depth: int) -> tuple[BestGameEverMade, int]:
    game = BestGameEverMade(
        [
            AI_Player(Negamax(depth)),
            AI_Player(Negamax(depth)),
        ]
    )
    hist = game.play(verbose=False)

    last_game_state = hist[-1]
    # Last element is the final game state
    assert isinstance(last_game_state, BestGameEverMade)

    return last_game_state, len(hist) - 1


def time_function[Ret](fun: Callable[[], Ret]) -> tuple[Ret, float]:
    start = time.time()
    result = fun()
    end = time.time()
    return result, end - start


def main() -> None:
    for depth in range(1, 9):
        print(f"Running game with Negamax({depth=})...")
        (game, total_moves), time_taken = time_function(lambda: run_game(depth))
        status_parts = [
            f"Time taken: {time_taken:.2f} seconds",
            f"Total moves: {total_moves}",
            f"Player 1: {game.player_1_score} points, {game.player_1_remaining_tokens} moves left",
            f"Player 2: {game.player_2_score} points, {game.player_2_remaining_tokens} moves left",
        ]
        print("\n".join(status_parts))
        print()


if __name__ == "__main__":
    main()

from typing import Any

from easyAI.TwoPlayerGame import TwoPlayerGame

from main import Move

class Player:
    name: str

    def ask_move(self, game: TwoPlayerGame[Move]) -> Move: ...

class Human_Player(Player):
    def __init__(self, name: str = ...) -> None: ...

class AI_Player(Player):
    def __init__(self, AI_algo: Any, name: str = ...) -> None: ...  # pyright: ignore[reportAny, reportExplicitAny]

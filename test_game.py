# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from game import BOARD_COLUMNS, BOARD_ROWS, BestGameEverMade, BoardElement

if TYPE_CHECKING:
    from easyAI.TwoPlayerGame import PlayerIndex


def _make_column(*elements: str) -> list[BoardElement]:
    return [BoardElement(el) for el in elements]


def test_column_is_full_with_empty_column() -> None:
    column: list[BoardElement] = []
    assert BestGameEverMade._column_is_full(column) is False


def test_column_is_full_single_element() -> None:
    column = [BoardElement.BLACK]
    assert BestGameEverMade._column_is_full(column) is False


def test_column_is_full_with_all_elements() -> None:
    column = [BoardElement.BLACK for _ in range(BOARD_ROWS)]
    assert BestGameEverMade._column_is_full(column) is True


def test_possible_moves_clean_game() -> None:
    """Should return all possible moves when the game hasn't started"""

    game = BestGameEverMade([])
    assert game.possible_moves() == [1, 2, 3, 4, 5, 6, 7]


def test_possible_moves_skips_full_column() -> None:
    """Should remove full columns from possible moves"""

    game = BestGameEverMade([])
    game.board[0] = [BoardElement.BLACK for _ in range(BOARD_ROWS)]
    assert game.possible_moves() == [2, 3, 4, 5, 6, 7]  # "1" should be removed


def test_possible_moves_skips_multiple_full_columns() -> None:
    """Should remove full columns from possible moves"""

    game = BestGameEverMade([])
    game.board[0] = [BoardElement.BLACK for _ in range(BOARD_ROWS)]
    game.board[4] = [BoardElement.BLACK for _ in range(BOARD_ROWS)]
    assert game.possible_moves() == [2, 3, 4, 6, 7]  # "1" and "5", should be removed


def test_possible_moves_all_columns_full() -> None:
    """Should return an empty list when all columns are full"""

    game = BestGameEverMade([])
    for row in range(BOARD_COLUMNS):
        game.board[row] = [BoardElement.BLACK for _ in range(BOARD_ROWS)]
    assert game.possible_moves() == []


@pytest.mark.parametrize(
    ("initial_player", "expected_element"),
    [
        (1, BoardElement.WHITE),
        (2, BoardElement.BLACK),
    ],
)
def test_current_player_board_element(
    initial_player: PlayerIndex, expected_element: BoardElement
) -> None:
    """Should return WHITE for player 1 and BLACK for player 2"""

    game = BestGameEverMade([], initial_player)
    assert game._current_player_board_element == expected_element


@pytest.mark.parametrize(
    "column",
    [
        _make_column("w"),
        _make_column("w", "b"),
    ],
)
def test_find_tokens_to_capture_too_low(column: list[BoardElement]) -> None:
    """Should return None when there are too little tokens below the given index"""

    game = BestGameEverMade([])
    below = len(column) - 1
    assert game._find_tokens_to_capture(column, below) is None


@pytest.mark.parametrize(
    "column",
    [
        _make_column("w", "b", "w"),
        _make_column("b", "w", "b", "w"),
        _make_column("b", "w", "b", "b"),
    ],
)
def test_find_tokens_to_capture_no_consecutive_tokens(
    column: list[BoardElement],
) -> None:
    """
    Should return None when there are no opponent tokens
    to capture below the given index
    """

    game = BestGameEverMade([])
    below = len(column) - 2
    assert game._find_tokens_to_capture(column, below) is None


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (_make_column("b", "b", "w"), [0, 1]),
        (_make_column("w", "b", "b", "w"), [1, 2]),
    ],
)
def test_find_tokens_to_capture_consecutive_tokens(
    column: list[BoardElement], expected: list[int]
) -> None:
    """
    Should return captureable indexes when there are 2 consecutive
    opponent tokens below the given index
    """

    game = BestGameEverMade([])
    game.board[0] = []
    below = len(column) - 1
    assert game._find_tokens_to_capture(column, below) == expected


def test_find_tokens_to_score_empty_column() -> None:
    """Should return None when the column is empty"""

    game = BestGameEverMade([])
    game.board[0] = []
    assert game._find_tokens_to_score(game.board[0]) is None


def test_find_tokens_to_score_no_tokens_to_score() -> None:
    """Should return None when there are no consecutive tokens to score"""

    game = BestGameEverMade([])
    column = _make_column("w", "b", "w", "b")
    assert game._find_tokens_to_score(column) is None


def test_find_tokens_to_score_three_tokens_to_score() -> None:
    """Should return amount of scored tokens when there are three tokens to capture"""

    game = BestGameEverMade([])
    column = _make_column("w", "w", "w")
    assert game._find_tokens_to_score(column) == 3


def test_find_tokens_to_score_four_tokens_to_score() -> None:
    """
    Should return amount of scored tokens when there are four tokens to capture

    This can happen after a successful capture of 2 BLACK tokens:
    w, b, b, w -> w, w, w, w
    """

    game = BestGameEverMade([])
    column = _make_column("w", "w", "w", "w")
    assert game._find_tokens_to_score(column) == 4


def test_find_tokens_to_score_full_column_with_no_tokens_to_score() -> None:
    """Should return None when a column is full, but it has no tokens to score"""

    game = BestGameEverMade([])
    column = _make_column("w", "b", "w", "b", "w")
    assert game._find_tokens_to_score(column) is None

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from game import (
    BOARD_COLUMNS,
    BOARD_ROWS,
    BestGameEverMade,
    BoardElement,
    Column,
    Point,
)

if TYPE_CHECKING:
    from easyAI.TwoPlayerGame import PlayerIndex


def _make_column(*elements: str) -> Column:
    return [BoardElement(el) for el in elements]


def test_column_is_full_with_empty_column() -> None:
    assert BestGameEverMade._column_is_full([]) is False


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
def test_find_tokens_to_capture_too_low(column: Column) -> None:
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
def test_find_tokens_to_capture_no_consecutive_tokens(column: Column) -> None:
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
    column: Column, expected: list[int]
) -> None:
    """
    Should return captureable indexes when there are 2 consecutive
    opponent tokens below the given index
    """

    game = BestGameEverMade([])
    below = len(column) - 1
    assert game._find_tokens_to_capture(column, below) == expected


def test_find_tokens_to_score_empty_board() -> None:
    """Should return an empty set when the board is empty"""

    game = BestGameEverMade([])
    assert game._find_tokens_to_score() == set()


def test_find_tokens_to_score_no_tokens_to_score_in_column() -> None:
    """Should return an empty set when there are no tokens to score in a column"""

    game = BestGameEverMade([])
    game.board[0] = _make_column("w", "b", "w", "b")
    assert game._find_tokens_to_score() == set()


def test_find_tokens_to_score_no_tokens_to_score_in_row() -> None:
    """Should return an empty set when there are no tokens to score in a row"""

    game = BestGameEverMade([])
    game.board[0] = _make_column("w")
    game.board[1] = _make_column("b")
    game.board[2] = _make_column("w")
    game.board[3] = _make_column("b")
    assert game._find_tokens_to_score() == set()


def test_find_tokens_to_score_in_a_column_interrupted_by_other_player() -> None:
    """
    Should return an empty set when there are 3 current player tokens in a column,
    but they are interrupted by the opponent
    """

    game = BestGameEverMade([])
    game.board[0] = _make_column("w", "b", "w", "w")
    assert game._find_tokens_to_score() == set()


def test_find_tokens_to_score_in_a_row_interrupted_by_other_player() -> None:
    """
    Should return an empty set when there are 3 current player tokens in a row,
    but they are interrupted by the opponent
    """

    game = BestGameEverMade([])
    game.board[0] = _make_column("w")
    game.board[1] = _make_column("b")
    game.board[2] = _make_column("w")
    game.board[3] = _make_column("w")
    assert game._find_tokens_to_score() == set()


def test_find_tokens_to_score_in_a_row_interrupted_by_empty_column() -> None:
    """
    Should return an empty set when there are 3 current player tokens in a row,
    but they are interrupted by an empty column
    """

    game = BestGameEverMade([])
    game.board[0] = _make_column("w")
    game.board[1] = _make_column()
    game.board[2] = _make_column("w")
    game.board[3] = _make_column("w")
    assert game._find_tokens_to_score() == set()


def test_find_tokens_to_score_three_tokens_to_score_in_a_column() -> None:
    """Should return three points when there are three tokens in a column to capture"""

    game = BestGameEverMade([])
    game.board[0] = _make_column("w", "w", "w")
    assert game._find_tokens_to_score() == {(0, 0), (0, 1), (0, 2)}


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (
            # 3 consecutive tokens
            [
                _make_column("w"),
                _make_column("w"),
                _make_column("w"),
            ],
            {(0, 0), (1, 0), (2, 0)},
        ),
        (
            # 5 consecutive tokens - can happen after a capture
            [
                _make_column("w"),
                _make_column("w"),
                _make_column("w"),
                _make_column("w"),
                _make_column("w"),
            ],
            {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)},
        ),
        (
            # Breaking the consecutive sequence should still give the same result
            [
                _make_column("w"),
                _make_column("w"),
                _make_column("w"),
                _make_column("b"),
            ],
            {(0, 0), (1, 0), (2, 0)},
        ),
        (
            # 3 consecutive tokens, but in the 2nd row
            [
                _make_column("b", "w"),
                _make_column("b", "w"),
                _make_column("b", "w"),
            ],
            {(0, 1), (1, 1), (2, 1)},
        ),
    ],
)
def test_find_tokens_to_score_consecutive_tokens_in_rows(
    columns: list[Column],
    expected: set[Point],
) -> None:
    """Should return three points when there are three tokens in a row to capture"""

    game = BestGameEverMade([])
    for col_idx, col in enumerate(columns):
        game.board[col_idx] = col
    assert game._find_tokens_to_score() == expected


def test_find_tokens_to_score_four_tokens_to_score_in_a_column() -> None:
    """
    Should return four points when there are four tokens to score in a column

    This can happen after a successful capture of 2 tokens:
    [W] <- was just placed by player 1
    [B]
    [B]
    [W]
    Would turn into:
    [W]
    [W] <- captured by token above
    [W] <- captured by token above
    [W]
    Which can then be scored as 4 tokens at the same time
    """

    game = BestGameEverMade([])
    game.board[0] = _make_column("w", "w", "w", "w")
    assert game._find_tokens_to_score() == {(0, 0), (0, 1), (0, 2), (0, 3)}


def test_find_tokens_to_score_from_row_and_column_after_capture() -> None:
    """
    Should return six points when there is a 4-long row after a capture of 2 tokens

    This can happen after a successful capture of 2 tokens:
        v----------- [W] was just placed by player 1
    [ ][W][ ][ ]
    [B][B][W][B]
    [W][B][W][W]
    Would turn into:
        v----------- whole column got turned into [W] after the capture
    [ ][W][ ][ ]
    [B][W][W][B]
    [W][W][W][W]
    Which can then be scored as 6 tokens at the same time:
    - 4 from the bottom row
    - 3 from the column (1 overlaps)
    """

    game = BestGameEverMade([])
    game.board[0] = _make_column("w", "b")
    game.board[1] = _make_column("w", "w", "w")
    game.board[2] = _make_column("w", "w")
    game.board[3] = _make_column("w", "b")
    to_score_from_column = {(1, 0), (1, 1), (1, 2)}
    to_score_from_row = {(0, 0), (1, 0), (2, 0), (3, 0)}
    assert game._find_tokens_to_score() == to_score_from_column | to_score_from_row


def test_find_tokens_to_score_full_column_with_no_tokens_to_score() -> None:
    """Should return an empty set when a column is full, but has no tokens to score"""

    game = BestGameEverMade([])
    game.board[0] = _make_column("w", "b", "w", "b", "w")
    assert game._find_tokens_to_score() == set()


def test_find_tokens_to_score_full_row_with_no_tokens_to_score() -> None:
    """Should return an empty set when a row is full, but has no tokens to score"""

    game = BestGameEverMade([])
    game.board[0] = _make_column("w")
    game.board[1] = _make_column("b")
    game.board[2] = _make_column("w")
    game.board[3] = _make_column("b")
    game.board[4] = _make_column("w")
    assert game._find_tokens_to_score() == set()

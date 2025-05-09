"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    countX = 0
    countY = 0
    for row in board:
        for state in row:
            if state == X:
                countX += 1
            elif state == O:
                countY += 1
    
    if countX > countY:
        return O
    else:
        return X

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    if terminal(board):
        return None

    actionsSet = set()

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                actionsSet.add((i, j))

    return actionsSet

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action

    if board[i][j] is not None:
        raise ValueError

    newBoard = copy.deepcopy(board)

    newBoard[i][j] = player(board)

    return newBoard


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != '':
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != '':
            return board[0][i]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != '':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != '':
        return board[0][2]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True

    for row in board:
        for state in row:
            if state == EMPTY:
                return False

    return True

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    # Assuming that utility will only be called on a board if terminal(board) is True.

    result = winner(board)
    if result == X:
        return 1
    elif result == O:
        return -1
    else:
        return 0

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    if terminal(board):
        return None

    def max(board):
        answer = ()
        if terminal(board):
            return utility(board), answer
        else:
            temp = -1
            for action in actions(board):
                util = min(result(board, action))[0]
                if util > temp:
                    answer = action
                    temp = util
            return temp, answer

    def min(board):
        answer = ()
        if terminal(board):
            return utility(board), answer
        else:
            temp = 1
            for action in actions(board):
                util = max(result(board, action))[0]
                if util < temp:
                    answer = action
                    temp = util
            return temp, answer
        
    curPlayer = player(board)

    if curPlayer == X:
        return max(board)[1]
    else:
        return min(board)[1]
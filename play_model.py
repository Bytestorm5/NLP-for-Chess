import numpy as np
from keras.models import load_model
import chess
import embed_games_by_state as state_embeddings


letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
numbers = ['1', '2', '3', '4', '5', '6', '7', '8']

pieces = ['', 'p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']

def embed_to_move(move_embed):
    output = ''
    output += letters[np.argmax(move_embed[:8])]
    output +=  numbers[np.argmax(move_embed[8:16])]
    output += letters[np.argmax(move_embed[16:24])]
    output +=  numbers[np.argmax(move_embed[24:32])]
    output +=  pieces[np.argmax(move_embed[32:])]
    return output

def actual_move_from_embed(move_embed, board: chess.Board):
    move_embeddings = np.array([state_embeddings.embed_move(move.uci()) for move in board.generate_legal_moves()])
    input_vector = np.array(move_embed)

    distances = np.linalg.norm(move_embeddings - input_vector, axis=1)
    closest_index = np.argmin(distances)
    closest_vector = move_embeddings[closest_index]
    return embed_to_move(closest_vector)

def embed_to_board(board_embed):
    board = chess.Board('8/8/8/8/8/8/8/8 w - - 0 1')
    accumulator = 0
    for i in range(64):
        piece = pieces[np.argmax(board_embed[accumulator:accumulator+13])]
        accumulator += 13
        if piece != '':
            board.set_piece_at(i, chess.Piece.from_symbol(piece))
    
    assert accumulator == 832

    passant_square = np.argmax(board_embed[accumulator: accumulator+65]) - 1
    accumulator += 65
    if passant_square > -1:
        board.ep_square = passant_square
    
    assert accumulator == 897

    color = 1 - np.argmax(board_embed[accumulator:accumulator+2])
    board.turn = bool(color)
    accumulator += 2

    castle_fen = ''
    castle_symbols = 'KQkq'
    for i in range(4):
        if board_embed[accumulator] > 0.5:
            castle_fen += castle_symbols[i]
        accumulator += 1
    
    board.set_castling_fen(castle_fen)

    assert accumulator == len(board_embed)

    return board

# Load the saved model
model = load_model('model')

board = chess.Board()

while not board.is_game_over():
    if board.turn:
        # Human
        move_uci = input("UCI: ")
        move = chess.Move.from_uci(move_uci)
        if move in list(board.generate_legal_moves()):
            print(f' - User played {move_uci}')
            board.push(move)
        else:
            print(' - Illegal Move! Try again.')
            continue
    else:
        move_embed = model.predict(np.array([state_embeddings.embed_board(board)]))[0]
        move = chess.Move.from_uci(actual_move_from_embed(move_embed, board))
        board.push(move)
        print(f' - Model played {move.uci()}')    
        print(board)
# # Perform inference on the data subset
# predictions = model.predict(subset_embeddings)

# # Print the predictions
# for result in zip(subset_embeddings, subset_moves, predictions):
#     board = embed_to_board(result[0])
#     print(board)
#     print(embed_to_move(result[1]))
#     print()
#     print(actual_move_from_embed(result[2], board))
#     print('\n---------\n')
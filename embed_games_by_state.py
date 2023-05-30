import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM
import chess.pgn as pgn
import chess
import io
from tqdm import tqdm

# Define position vocabulary
positions = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', '1', '2', '3', '4', '5', '6', '7', '8']

pieces = ['', 'p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
promotion_pieces = ['n', 'b', 'r', 'q', 'P', 'N', 'B', 'R']

def embed_color(board):
    return np.array([1, 0]) if board.turn else np.array([0, 1])

def embed_board(board: chess.Board):
    encodings = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        embed = np.zeros(len(pieces))
        if piece is not None:
            embed[pieces.index(piece.symbol())] = 1
        else:
            embed[0] = 1
        encodings.append(embed)

    #encodings.append()

    ep: chess.Square = board.ep_square
    if ep is None:
        encodings.append(np.zeros(65))
    else:
        passant = np.zeros(65)
        passant[ep+1] = 1
        encodings.append(passant)  

    encodings.append(embed_color(board))

    encodings.append([int(board.has_kingside_castling_rights(board.turn))])
    encodings.append([int(board.has_queenside_castling_rights(board.turn))])
    encodings.append([int(board.has_kingside_castling_rights(not board.turn))])
    encodings.append([int(board.has_queenside_castling_rights(not board.turn))])

    return np.concatenate(encodings)

def embed_move(move_uci):       
    from_encoding = np.zeros(len(positions))
    from_encoding[positions.index(move_uci[0])] = 1
    from_encoding[positions.index(move_uci[1])] = 1

    to_encoding = np.zeros(len(positions))
    to_encoding[positions.index(move_uci[2])] = 1
    to_encoding[positions.index(move_uci[3])] = 1

    promote_encoding = np.zeros(4)
    if len(move_uci) > 4:
        promote_encoding[promotion_pieces.index(move_uci[4])] = 1

    return np.concatenate([from_encoding, to_encoding, promote_encoding])

def move_str(move: chess.Move, board: chess.Board):
    if move in [chess.Move.from_uci("e1g1"), chess.Move.from_uci("e1c1"),
                chess.Move.from_uci("e8g8"), chess.Move.from_uci("e8c8")]:
        san = board.san(move)
        if 'O' in san:
            return san
        else:
            return move.uci()
    else:
        return move.uci()

if __name__ == "__main__":
    X = []
    Y = []

    with open('games.txt', "r") as file:
        PGNs = file.read().split('\n\n\n')
        for PGN in tqdm(PGNs):
            game = pgn.read_game(io.StringIO(PGN))        
            #moves = game.mainline_moves().start

            prompt = []
            prev: chess.Move = None
            board = chess.Board()
            try:
                for move in game.mainline_moves():
                    if prev is not None:
                        board.pop()
                        prompt.append(move_str(prev, board))
                        X.append(embed_board(board))
                        board.push(prev)
                    
                        
                        Y.append(embed_move(prev.uci()))

                    prev = move
                    board.push(move)
            except AssertionError:
                # Fails at variants, but while positions are valid we can get something out of it
                pass
            
    np.save('X.npy', np.array(X), allow_pickle=True)
    np.save('Y.npy', np.array(Y), allow_pickle=True)

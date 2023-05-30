from chessdotcom import Client
import chessdotcom
import chess
from tqdm import tqdm


Client.request_config["headers"]["User-Agent"] = (
    "Data collection app for ML"
    "Contact me at gbytestorm@gmail.com"
)

archives = chessdotcom.get_player_game_archives('magnuscarlsen').json['archives']

with open('games.txt', 'w') as writer:
    for archive in tqdm(archives):
        url_split = archive.split('/')
        pgns = chessdotcom.get_player_games_by_month_pgn('magnuscarlsen', url_split[-2], url_split[-1]).text
        writer.write(pgns)
        writer.write('\n')
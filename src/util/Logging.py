

import logging



def setup_logging(log_folder = '.') -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] [%(name)s]: %(message)s',
        handlers=[
            logging.FileHandler(f'{log_folder}/app.log'),
            logging.StreamHandler()
        ]
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
import logging
from src.utils.logger import setup_logging
from src.models.train import train_func

setup_logging()
logger = logging.getLogger(__name__)


def main():
    logger.info("Hello from open-food-mlops!")
    train_func()


if __name__ == "__main__":
    main()

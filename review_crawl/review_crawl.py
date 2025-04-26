import time

from traceback import format_exc


from selenium.webdriver import (
    Firefox,
    FirefoxOptions
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from utils import (
    get_place_meta,
    get_review_ratings,
    get_review_texts,
    scroll_reviews
)


from typing import Literal

# logging stuff
import log_setup # this just sets up logging, it's not meant to be used as a module
import logging
logger = logging.getLogger(__name__)


class NoConnectedDriver(Exception):
    """Exception raised when there is no initialized WebDriver"""

class ReviewCrawler():
    _REVIEW_BUTTON = "//button[@class='hh2c6 ' and contains(., 'Bài đánh giá')]"
    def __init__(self, language: str = 'vi', headless: bool = True):
        self.driver_opts = FirefoxOptions()
        self._driver = None

        self.driver_opts.set_preference("intl.accept_languages", language)
        if headless:
            self.driver_opts.add_argument('--headless')

    def open(self):
        logger.info('Initializing driver...')
        self._driver = Firefox(options=self.driver_opts)
        self._wait = WebDriverWait(self._driver, 20)
        logger.info('Driver initialized...')

    def crawl_from(self,
        url: str,
        num_reviews: int,
        load_timeout: float
    ) -> dict[Literal['address', 'price_range', 'reviews', 'ratings'], str]:
        if self._driver is None:
            raise NoConnectedDriver('No WebDriver have been initialized. Please check if you have opened a connection!')
        
        print(f'Now crawling from \033[34;4m{url}\033[0m')
        logger.info(f'Connecting to {url}')
        self._driver.get(url)


        print('Getting place overview...')
        meta_data = get_place_meta(self._wait)
        results = self._crawl_process(url, load_timeout, num_reviews)
        if results is None:
            print('\033[31;1mCrawling unsuccesful! Check logs for details...\033[0m')
        else:
            print('\033[32;1mCrawling finished!\033[0m')
        return {
            'address': meta_data['address'],
            'price_range': meta_data['price_range'],
            'reviews': results[0],
            'ratings': results[1]
        }

    def close(self,):
        self._driver.quit()

    
    def _crawl_process(self,
        url: str,
        load_timeout: float,
        num_reviews: int
    ) -> tuple[list[str], list[str]] | None:
        # go to reviews tab by clicking reviews button
        self._driver\
            .find_element(By.XPATH, ReviewCrawler._REVIEW_BUTTON)\
            .click()
        time.sleep(load_timeout)

        print('Begin crawling reviews')
        logger.info('Loading reviews...')
        scroll_reviews(self._driver, num_reviews, load_timeout)

        logger.info('Crawling reviews data...')
        try:
            reviews = get_review_texts(self._driver)
            ratings = get_review_ratings(self._driver)
        except:
            logger.critical(f"""Exception occured while trying to crawl reviews for {url}
{format_exc()}""")
            return
        return reviews, ratings
import time

from traceback import format_exc


from selenium.webdriver import (
    Firefox,
    FirefoxOptions
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from ggplace_review_crawler.utils import (
    get_place_meta,
    get_review_ratings,
    get_review_texts,
    scroll_reviews
)


from typing import Literal

# logging stuff
# import ggplace_review_crawler._log_setup # this just sets up logging, it's not meant to be used as a module
import logging
logger = logging.getLogger(__name__)


class NoConnectedDriver(Exception):
    """Exception raised when there is no initialized WebDriver"""

class ReviewCrawler:
    """
    Crawling reviews from Google Place using a Firefox web driver.
    Before opening a web driver, you can configure options through `self.driver_opts`.

    Keep in mind that this should be an option in a Firefox browser.
    """
    _REVIEW_BUTTON = "//button[@class='hh2c6 ' and contains(., 'Bài đánh giá')]"
    def __init__(self, language: str = 'vi', headless: bool = True):
        """
        Initialize the crawler. You can specify the language for the browser
        as well as whether the browser should run in the background.

        :param str language: The default language of the browser. This should follow
            the ISO-639 languge code. This is `vi` by default.
        :param bool headless: Whether or not the web driver should run in the background.
            If `False`, a browser instance will appear on your taskbar.
        """
        self.driver_opts = FirefoxOptions()
        self._driver = None

        self.driver_opts.set_preference("intl.accept_languages", language)
        if headless: self.driver_opts.add_argument('--headless')

    def open(self):
        """
        Open a web driver instance with the pre-configured options.
        """
        logger.info('Initializing driver...')
        self._driver = Firefox(options=self.driver_opts)
        self._wait = WebDriverWait(self._driver, 20)
        logger.info('Driver initialized...')

    def crawl_from(self,
        url: str,
        num_reviews: int = 10,
        load_timeout: float = 2
    ) -> dict[Literal['address', 'price_range', 'reviews', 'ratings'], str]:
        """
        Starting crawling data from a place specified with `url`.

        :param int num_reviews: Specify how many reviews to crawl. To crawl all available
            reviews, set to `0`.
        :param float load_timeout: Specify the duration of timeout to wait between each interactions.
            This is very crucial when loading reviews and will result in inaccurate results if the waiting
            time is too low.
        
        :return: A dictionary containing `address`, `price_range`, `reviews` and their corresponding `ratings`.
        :rtype: dict
        """
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
        """
        Close the web driver.
        """
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
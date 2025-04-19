import math
import time

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC


from typing import Literal

import logging
logger = logging.getLogger(__name__)


_ADDRESS = By.CLASS_NAME, 'rogA2c '
_STAR = By.CLASS_NAME, "kvMYJc"

_REVIEW = By.XPATH, "//span[@class='wiI7pd']"
_PRICE = By.XPATH, '//div[@class="MNVeJb eXOdV eF9eN PnPrlf"]/div'
_MORE_BUTTON = By.XPATH, "//button[text()='Thêm']"
_SCROLLER = By.XPATH, "//div[@class='m6QErb DxyBCb kA9KIf dS8AEf XiKgde ']"

def get_place_meta(waiting_driver) -> dict[Literal['address', 'price_range'], str]:
    """"""
    try:
        price_range = waiting_driver.until(EC.presence_of_element_located(_PRICE))
        price_range = ''
    else:
        # If price_range exists, parse the text
        # There will be a case where price_range.text return empty string
        # This is because it is NOT in view, we use ActionChains to scroll into view
        if not price_range.text:
            x, y = price_range.location_once_scrolled_into_view.values()
            ActionChains(waiting_driver._driver)\
                .move_to_element_with_offset(price_range, x, y)\
                .perform()
            # price_range = waiting_driver.until(EC.presence_of_element_located(_PRICE))
        price_range = price_range.text.split('\n')[0]

    return {
        "address": waiting_driver.until(EC.presence_of_element_located(_ADDRESS))\
                                 .text,
        "price_range": price_range
    }
    
def scroll_reviews(
    driver,
    max_num_reviews: int | None = None,
    load_timeout: float = 2,
) -> None:
    """
    Scroll along the reviews tab to load more reviews. This function scrolls
    until no more reviews can be loaded or the maximum number of reviews loaded
    has been reached.

    :param WebDriver driver: A WebDriver instance connected to a place on Google Map.
    :param int | None max_num_reviews: The maximum number of reviews to load. If `None`,
        the function scrolls until no more reviews are loaded. This is `None` by default.
    :param float load_timeout: The timeout (seconds) between each scroll. This depends a lot
        on your connection speed and might give inaccurate results if not chosen properly.
    """
    # Note to self: the div class is a scrollable div. Sending the END key
    # to this div will scroll to bottom, after which more reviews are loaded.
    if max_num_reviews is None:
        max_num_reviews = math.inf
    scrollable_zone = driver.find_element(*_SCROLLER)

    cur_num_reviews = len(driver.find_elements(*_REVIEW))
    prev_num_reviews = cur_num_reviews - 1
    while (prev_num_reviews != cur_num_reviews) and cur_num_reviews < max_num_reviews:
        scrollable_zone.send_keys(Keys.END)

        temp = len(driver.find_elements(*_REVIEW))
        temp_prev = temp - 1
        while temp <= cur_num_reviews and temp != temp_prev:
            time.sleep(load_timeout) # wait to load
            temp_prev = temp
            temp = len(driver.find_elements(*_REVIEW))

        prev_num_reviews = cur_num_reviews
        cur_num_reviews = len(driver.find_elements(*_REVIEW))
        logger.debug(f'Found {cur_num_reviews} reviews during scrolling')

def get_review_texts(driver) -> list[str]:
    """
    Get the text of currently loaded reviews.

    :param WebDriver driver: A WebDriver instance connected to a place on Google Map.
    :rtype: list[str]
    :return: A list of review texts.
    """
    # fully expand review texts by cliking more button
    for more_button in driver.find_elements(*_MORE_BUTTON):
        more_button.click()
    return list(review.text\
                for review in driver.find_elements(*_REVIEW))

def get_review_ratings(driver) -> list[str]:
    """
    Get the star rating of currently loaded reviews.

    :param WebDriver driver: A WebDriver instance connected to a place on Google Map.
    :rtype: list[str]
    :return: A list of review rating as strings.
    """
    return list(star_span.get_attribute('aria-label')[0]\
                for star_span in driver.find_elements(*_STAR))
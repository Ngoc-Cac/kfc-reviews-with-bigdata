import math
import time

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from typing import Literal


def get_place_meta(driver) -> dict[Literal['address', 'price_range'], str]:
    """"""
    return {
        "address": driver.find_element(By.CLASS_NAME, 'rogA2c ').text,
        "price_range": driver.find_element(By.XPATH, '//div[@class="MNVeJb eXOdV eF9eN PnPrlf"]')\
                             .text.split('\n')[0]
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
    scrollable_zone = driver.find_element(By.XPATH, "//div[@class='m6QErb DxyBCb kA9KIf dS8AEf XiKgde ']")

    cur_num_reviews = len(driver.find_elements(By.CLASS_NAME, "wiI7pd"))
    prev_num_reviews = cur_num_reviews - 1
    while (prev_num_reviews != cur_num_reviews) and cur_num_reviews < max_num_reviews:
        scrollable_zone.send_keys(Keys.END)
        time.sleep(load_timeout) # wait to load

        prev_num_reviews = cur_num_reviews
        cur_num_reviews = len(driver.find_elements(By.CLASS_NAME, "wiI7pd"))

def get_review_texts(driver) -> list[str]:
    """
    Get the text of currently loaded reviews.

    :param WebDriver driver: A WebDriver instance connected to a place on Google Map.
    :rtype: list[str]
    :return: A list of review texts.
    """
    # fully expand review texts by cliking more button
    for more_button in driver.find_elements(By.XPATH, "//button[text()='Thêm']"):
        more_button.click()
    return list(review.text for review in driver.find_elements(By.CLASS_NAME, "wiI7pd"))

def get_review_ratings(driver) -> list[str]:
    """
    Get the star rating of currently loaded reviews.

    :param WebDriver driver: A WebDriver instance connected to a place on Google Map.
    :rtype: list[str]
    :return: A list of review rating as strings.
    """
    return list(star_span.get_attribute('aria-label')[0]\
                for star_span in driver.find_elements(By.CLASS_NAME, "kvMYJc"))
# Big-Data-G4: Review Crawler
Google Maps reviews crawling process for the main project. The crawler uses [selenium](https://pypi.org/project/selenium/) to simulate the website and crawl data through HTML elements.\
Please note that while Google Maps **DO** provide an API, this requires you to set up a billing account and thus can be inconvenient.

## How do I run the crawler for my places?
1. Firstly, be sure to have all necessary libraries installed, mentioned in [`requirements.txt`](/Big-Data-G4/requirements.txt). Run the following snippet for quick installation:
```
pip install -r requirements.txt
```
2. Create your own scripts using the pre-configured `ggplace_review_crawler.ReviewCrawler` to crawl reviews from your place of choice. The crawler will only return the results as a Python `dict`. You will need to explicitly define how you want to save the results. Check out [`crawling.ipynb`](./crawling.ipynb) for an example.

If any crawling was unsuccessful, please create an Issue and provide the Exception traceback (if any). Traceback is available in `/logs`.
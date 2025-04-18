# Big-Data-G4
*This branch is **NOT** meant to be merged into `/main`*

Google Maps reviews crawling process for the main project. The crawler uses [selenium](https://pypi.org/project/selenium/) to simulate the website and crawl data through HTML elements.\
Please note that while Google Maps **DO** provide an API, this requires you to set up a billing account and thus can be inconvenient.

Because reviews contain a multitude of UTF-8 encoded characters, which will be painful to store as raw strings in any character delimited file, we store each review as a base-64 encoded string instead of raw strings. Thus, the review is **not human-readable** unless one decodes it.

## How do I run the crawler for my places?
1. Firstly, be sure to have all necessary libraries installed, mentioned in [`requirements.txt`](/Big-Data-G4/requirements.txt). Run the following snippet for quick installation:
```
pip install -r requirements.txt
```
2. Secondly, you need to list out places you want to crawl reviews from. Please list all links in a file **seperated** by newline characters, see [`links.txt`](/Big-Data-G4/links.txt) for an example.

3. Then, go to [`crawler.conf`](/Big-Data-G4/crawler.conf) and replace the `file_location` with the path to the file where you have listed out the links of interest. You may also change other attributes if necessary.

Now, navigate to [`review_crawl.py`](/Big-Data-G4/review_crawl.py) and run the file as is.

If any crawling was unsuccessful, please create an Issue and provide the Exception traceback (if any). Traceback Exception is available in `/logs`
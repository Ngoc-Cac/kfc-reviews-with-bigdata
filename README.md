# Big-Data-G4
*This branch is **NOT** meant to be merged into `/main`*

Google Maps reviews crawling process for the main project. The crawler uses [selenium](https://pypi.org/project/selenium/) to simulate the website and crawl data through HTML elements.

Because reviews contain a multitude of UTF-8 encoded characters, which will be painful to store as raw strings in any character delimited file, we store each review as a base-64 encoded string instead of raw strings. Thus, the review is **not human-readable** unless one decodes it.
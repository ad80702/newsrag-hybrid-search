import requests
import os
from dotenv import load_dotenv

load_dotenv()  # reads your .env file so the API key isn't hardcoded in your script

API_KEY = os.getenv("GUARDIAN_API_KEY")
BASE_URL = "https://content.guardianapis.com/search"

def fetch_articles(query, page_size=10):
    params = {
        "q": query,                 # the search term, e.g. "federal reserve"
        "api-key": API_KEY,
        "page-size": page_size,     # how many articles to return
        "show-fields": "bodyText",  # this tells the API: give me the full article text, not just headline
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    articles = []
    for item in data["response"]["results"]:
        articles.append({
            "title": item["webTitle"],
            "url": item["webUrl"],
            "text": item["fields"]["bodyText"],
        })

    return articles


if __name__ == "__main__":
    results = fetch_articles("artificial intelligence")
    for a in results:
        print(a["title"])
        print(a["text"][:200])  # print first 200 characters just to check it worked
        print("---")
import requests
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

load_dotenv()

# Set your OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def extract_links(url):
    """Extracts all links from a webpage."""
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to fetch the page: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    main_content = soup.find("main") or soup.body  # Prefer 'main' content
    links = [a['href'] for a in main_content.find_all('a', href=True)]
    
    return links

def filter_links_with_llm(links):
    """Filters relevant links using OpenRouter's LLM with improved error handling."""
    if not links:
        return []

    prompt = f"""
    Here is a list of links extracted from a webpage:
    {json.dumps(links, indent=4)}

    Your task: 
    - Only return links related to rebate programs, grants, loans, or energy efficiency initiatives.
    - Exclude links from navigation bars, footers, or unrelated sections.

    Return the filtered list as a valid JSON array.
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "google/gemini-2.0-flash-thinking-exp-1219:free",  # Change model if needed
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        response_json = response.json()
        
        # Print response for debugging
        print("API Response:", json.dumps(response_json, indent=4))
        
        if "choices" in response_json:
            try:
                filtered_links = json.loads(response_json["choices"][0]["message"]["content"])
                return filtered_links
            except (json.JSONDecodeError, IndexError, KeyError):
                print("Error decoding response content.")
                return []
        else:
            print("Unexpected API response format. 'choices' key missing.")
            return []
    else:
        print(f"Failed to process LLM request: {response.status_code}")
        print("Error response:", response.text)  # Print API error message
        return []

def save_links_to_json(links, filename="filtered_links.json"):
    """Saves the extracted links to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(links, f, indent=4)

if __name__ == "__main__":
    url = "https://natural-resources.canada.ca/energy-efficiency/homes/canada-greener-homes-initiative/24831"

    all_links = extract_links(url)
    relevant_links = filter_links_with_llm(all_links)

    if relevant_links:
        print(f"Extracted {len(relevant_links)} relevant links via OpenRouter.")
        save_links_to_json(relevant_links)
        print("Relevant links saved to filtered_links.json")
    else:
        print("No relevant links found.")

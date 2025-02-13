import requests
from bs4 import BeautifulSoup

# URL of Codeforces Problem (replace with an actual problem URL)
problem_url = "https://codeforces.com/problemset/problem/1/A"

# Fetch problem page
response = requests.get(problem_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Check the HTML structure to identify where the solutions are stored
print(soup.prettify())  # This will print the entire HTML structure, inspect the code


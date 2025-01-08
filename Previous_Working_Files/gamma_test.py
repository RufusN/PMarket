import requests

response = requests.get("https://gamma-api.polymarket.com/markets")
markets = response.json()

# Example: Print the first market's data
print(markets[0])
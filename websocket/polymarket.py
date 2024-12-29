from gql import Client, gql
from gql.transport.websockets import WebsocketsTransport

# Set up the WebSocket transport
transport = WebsocketsTransport(
    url='wss://api.thegraph.com/subgraphs/name/polymarket/polymarket-matic'
)

# Create the client with the defined transport
client = Client(transport=transport, fetch_schema_from_transport=True)

# Define your subscription query
subscription = gql("""
    subscription {
        markets(first: 5) {
            id
            question
            outcomes {
                id
                price
                name
            }
        }
    }
""")

# Execute the subscription
async def main():
    async with client as session:
        async for result in session.subscribe(subscription):
            print("Received update:")
            print(result)

import asyncio
asyncio.run(main())

# File: polymarket_ws.py

import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    print("Received data:")
    print(data)

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket connection opened")
    # Send subscription message if required
    # ws.send(json.dumps({"type": "subscribe", "channel": "live_markets"}))

websocket.enableTrace(True)

# Replace with the actual websocket URL
websocket_url = 'https://clob.polymarket.com/'

ws = websocket.WebSocketApp(websocket_url,
                            on_open=on_open,
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close)

ws.run_forever()

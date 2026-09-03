import asyncio
import websockets


async def main():
    uri = "ws://127.0.0.1:8000/ws/threat-feed"

    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as websocket:
        print("WebSocket connected.")
        print("Waiting for a live threat alert...")

        message = await websocket.recv()

        print("\nLIVE THREAT RECEIVED:")
        print(message)


if __name__ == "__main__":
    asyncio.run(main())

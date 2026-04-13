from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clients = []


@app.post("/sns")
async def receive_sns(request: Request):
    data = await request.json()

    if data.get("Type") == "SubscriptionConfirmation":
        subscribe_url = data.get("SubscribeURL")

        if subscribe_url:
            async with httpx.AsyncClient() as client:
                await client.get(subscribe_url)

            return {"message": "SNS subscription confirmed successfully"}

        return {"message": "SubscriptionConfirmation received, but SubscribeURL missing"}

    dead_clients = []

    for queue in clients:
        try:
            await queue.put(data)
        except Exception:
            dead_clients.append(queue)

    for queue in dead_clients:
        if queue in clients:
            clients.remove(queue)

    return {"message": "SNS message received and sent to clients"}


@app.get("/events")
async def sse():
    queue = asyncio.Queue()
    clients.append(queue)

    async def event_generator():
        try:
            while True:
                data = await queue.get()
                yield f"data: {json.dumps(data)}\n\n"

        except asyncio.CancelledError:
            print("Client disconnected")

        finally:
            if queue in clients:
                clients.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/")
def health():
    return {"health": "The sse server is running actively at this url"}
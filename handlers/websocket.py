import asyncio

import aioredis
import async_timeout
import ujson
from sanic.request import Request
from websockets.exceptions import ConnectionClosed
import sanic.exceptions as exs
from loguru import logger


# ws
async def genreate(request: Request, ws):
    try:
        logger.info(f"received generate request")

        uid = request.args['mode'][0]
        room_name = request.args['room_name'][0]
        user_type = request.args['user_type'][0]

        while True:
            try:
                async with async_timeout.timeout(1):
                    message = await redis_pubsub.get_message(ignore_subscribe_messages=True)
                    if message is not None:
                        await ws.send(message['data'])
                        # if message["data"] == STOPWORD:
                        #     print("(Reader) STOP")
                        #     break
                    await asyncio.sleep(0.01)
            except asyncio.TimeoutError:
                pass
    except Exception:
        print('fatal error')
        raise ConnectionClosed(1010, "login again")
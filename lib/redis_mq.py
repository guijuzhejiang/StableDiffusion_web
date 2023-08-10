import asyncio
import datetime
import json
import math
import os
import socket
import time
import traceback
import uuid

import aioredis
import redis

from lib.common.common_util import logging, generate_random
from utils.global_vars import CONFIG


class RedisMQ:
    task_queue_name = f'{CONFIG["ocr"]["lang"]}_ocr_tq'
    group_name = f'{CONFIG["ocr"]["lang"]}_ocr_consumer_group'
    consumer_name = f'{CONFIG["ocr"]["lang"]}_ocr_consumer'
    host = 'localhost'
    port = 6379
    db = 11

    def __init__(self, host='localhost', port=6379, db=11):
        self.host = host
        self.port = port
        self.db = db

        self.redis_session = aioredis.from_url(f"redis://{host}:{port}", db=db, decode_responses=True)

    def sync_connect(self):
        return redis.from_url(f"redis://{self.host}:{self.port}", db=self.db, decode_responses=True)

    async def connect(self):
        self.redis_session = await aioredis.from_url(f"redis://{self.host}:{self.port}", db=self.db,
                                                     decode_responses=True)

    async def close(self):
        await self.redis_session.close()

    async def rpc_call(self, call_queue_name=None, reply_queue_name='reply_task', **kwargs):
        if call_queue_name is None:
            call_queue_name = self.task_queue_name

        reply_queue_name = f'{socket.gethostname()}' \
                           f'_{str(time.time()).replace(".", "_")}' \
                           f'_{os.getpid()}_{generate_random(5)}'

        try:
            # lock_id = await self.acquire_lock('mq_locker')

            task_id = await self.redis_session.xadd(call_queue_name, {'json_msg': json.dumps(kwargs),
                                                                      'reply_queue_name': reply_queue_name})

            # await self.release_lock('mq_locker', lock_id)

            res = await self.redis_session.xread({reply_queue_name: '0-0'},
                                                 block=CONFIG['server']['msg_expire_secs'] * 1000)
            # res = self.consume(reply_queue_name)
        except Exception:
            logging(
                f"[RedisMQ][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")
            return {'error': 'redis mq fatal error'}

        try:
            await self.redis_session.xdel(call_queue_name, task_id)
            await self.redis_session.delete(reply_queue_name)
        except Exception:
            logging(
                f"[RedisMQ][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")

        if not res:
            return {'error': 'redis mq fatal error'}
        else:
            return json.loads(res[0][1][0][1]['json_msg'])

    async def pub(self, queue_name, msg):
        try:
            stream_id = await self.redis_session.xadd(queue_name, msg)
        except Exception:
            logging(
                f"[RedisMQ][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")

    async def consume(self, queue_name=None):
        if queue_name is None:
            queue_name = self.task_queue_name
        try:
            await self.redis_session.xgroup_create(queue_name, self.group_name, id='0-0', mkstream=True)
        except Exception as e:
            if not "BUSYGROUP Consumer Group name already exists" in traceback.format_exc():
                logging(
                    f"[RedisMQ][{datetime.datetime.now()}]:"
                    f"{traceback.format_exc()}",
                    f"logs/error.log")  # BUSYGROUP Consumer Group name already exists  不能重复创建消费者组。
        while True:
            # redis服务端必须是5.0以上，并且确保这个键的类型是stream不能是list数据结构。
            results = await self.redis_session.xreadgroup(self.group_name, self.consumer_name,
                                                          {queue_name: ">"}, count=1, block=2 * 1000)
            if results:
                # self.logger.debug(f'从redis的 [{self._queue_name}] stream 中 取出的消息是：  {results}  ')
                # self._print_message_get_from_broker('redis', results)
                print(results[0][1])
                for msg_id, msg in results[0][1]:
                    yield (msg_id, msg)

            await asyncio.sleep(0.2)

    async def acquire_lock(self, lock_name, acquire_timeout=3, lock_timeout=2):
        """
        基于 Redis 实现的分布式锁

        self.redis_session: Redis 连接
        :param lock_name: 锁的名称
        :param acquire_timeout: 获取锁的超时时间，默认 3 秒
        :param lock_timeout: 锁的超时时间，默认 2 秒
        :return: identifier
        """

        identifier = str(uuid.uuid4())
        lockname = f'lock:{lock_name}'
        lock_timeout = int(math.ceil(lock_timeout))

        end = time.time() + acquire_timeout

        while time.time() < end:
            # 如果不存在这个锁则加锁并设置过期时间，避免死锁
            if await self.redis_session.set(lockname, identifier, ex=lock_timeout, nx=True):
                return identifier

            await asyncio.sleep(0.01)

        return False

    async def release_lock(self, lock_name, identifier):
        """
        释放锁

        self.redis_session: Redis 连接
        :param lockname: 锁的名称
        :param identifier: 锁的标识
        :return:
        """
        unlock_script = """
        if redis.call("get",KEYS[1]) == ARGV[1] then
            return redis.call("del",KEYS[1])
        else
            return 0
        end
        """
        lockname = f'lock:{lock_name}'
        unlock = self.redis_session.register_script(unlock_script)
        result = unlock(keys=[lockname], args=[identifier])
        if result:
            return True
        else:
            return False


class SyncRedisMQ:
    task_queue_name = f'task_q'
    group_name = f'{CONFIG["ocr"]["lang"]}_ocr_consumer_group'
    consumer_name = f'{CONFIG["ocr"]["lang"]}_ocr_consumer'
    host = 'localhost'
    port = 6379
    db = 11

    def __init__(self, host='localhost', port=6379, db=11, task_queue_name='task_q'):
        self.host = host
        self.port = port
        self.db = db
        self.task_queue_name = task_queue_name

        # setup redis session
        self.redis_session = redis.from_url(f"redis://{host}:{port}", db=db, decode_responses=True)

    def close(self):
        self.redis_session.close()

    def rpc_call(self, call_queue_name=None, reply_queue_name='reply_task', **kwargs):
        """

        Args:
            call_queue_name:
            reply_queue_name:
            **kwargs: 字典元素只能是str

        Returns:

        """
        if call_queue_name is None:
            call_queue_name = self.task_queue_name

        reply_queue_name = f'{socket.gethostname()}' \
                           f'_{str(time.time()).replace(".", "_")}' \
                           f'_{os.getpid()}_{generate_random(5)}'
        try:
            task_id = self.redis_session.xadd(call_queue_name, {'json_msg': json.dumps(kwargs),
                                                                'reply_queue_name': reply_queue_name})

            res = self.redis_session.xread({reply_queue_name: '0-0'}, block=CONFIG['server']['msg_expire_secs'] * 1000)
            # res = self.consume(reply_queue_name)
        except Exception:
            logging(
                f"[RedisMQ][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")
            return {'error': 'redis mq fatal error'}

        try:
            self.redis_session.xdel(call_queue_name, task_id)
            task_id = self.redis_session.delete(reply_queue_name)
        except Exception:
            logging(
                f"[RedisMQ][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")

        if not res:
            return {'error': 'fatal error'}
        else:
            return json.loads(res[0][1][0][1]['json_msg'])

    def pub(self, queue_name, msg, expire_secs=0):
        try:
            stream_id = self.redis_session.xadd(queue_name, msg)
        except Exception:
            logging(
                f"[RedisMQ][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")
        else:
            if expire_secs:
                self.redis_session.expire(queue_name, expire_secs)

    def consume(self, queue_name=None):
        if queue_name is None:
            queue_name = self.task_queue_name
        try:
            self.redis_session.xgroup_create(queue_name, self.group_name, id='0-0', mkstream=True)
        except Exception as e:
            if not "BUSYGROUP Consumer Group name already exists" in traceback.format_exc():
                logging(
                    f"[RedisMQ][{datetime.datetime.now()}]:"
                    f"{traceback.format_exc()}",
                    f"logs/error.log")  # BUSYGROUP Consumer Group name already exists  不能重复创建消费者组。
        while True:
            # redis服务端必须是5.0以上，并且确保这个键的类型是stream不能是list数据结构。
            results = self.redis_session.xreadgroup(self.group_name, self.consumer_name,
                                                    {queue_name: ">"}, count=1, block=60 * 1000)
            if results:
                # self.logger.debug(f'从redis的 [{self._queue_name}] stream 中 取出的消息是：  {results}  ')
                # self._print_message_get_from_broker('redis', results)
                print(results[0][1])
                for msg_id, msg in results[0][1]:
                    yield (msg_id, msg)

            time.sleep(0.2)

    def acquire_lock(self, lock_name, acquire_timeout=3, lock_timeout=2):
        """
        基于 Redis 实现的分布式锁

        self.redis_session: Redis 连接
        :param lock_name: 锁的名称
        :param acquire_timeout: 获取锁的超时时间，默认 3 秒
        :param lock_timeout: 锁的超时时间，默认 2 秒
        :return:
        """

        identifier = str(uuid.uuid4())
        lockname = f'lock:{lock_name}'
        lock_timeout = int(math.ceil(lock_timeout))

        end = time.time() + acquire_timeout

        while time.time() < end:
            # 如果不存在这个锁则加锁并设置过期时间，避免死锁
            if self.redis_session.set(lockname, identifier, ex=lock_timeout, nx=True):
                return identifier

            time.sleep(0.001)

        return False

    def release_lock(self, lock_name, identifier):
        """
        释放锁

        self.redis_session: Redis 连接
        :param lockname: 锁的名称
        :param identifier: 锁的标识
        :return:
        """
        unlock_script = """
        if redis.call("get",KEYS[1]) == ARGV[1] then
            return redis.call("del",KEYS[1])
        else
            return 0
        end
        """
        lockname = f'lock:{lock_name}'
        unlock = self.redis_session.register_script(unlock_script)
        result = unlock(keys=[lockname], args=[identifier])
        if result:
            return True
        else:
            return False


async def main():
    # init
    a = RedisMQ()
    results = await a.redis_session.xreadgroup(a.group_name, a.consumer_name,
                                         {'jp_OperatorOCR': ">"}, count=1, block=2 * 1000)
    print(results)
    return


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

import os

from sanic import Blueprint
from sanic import Sanic
from sanic_cors import CORS

from handlers.main import SDGenertae, SDHires
from lib.redis_mq import RedisMQ
from utils.global_vars import CONFIG

# Blueprint
bp = Blueprint("ai_tasks")
# add_route
bp.add_route(SDGenertae.as_view(), "/sd/generate")
bp.add_route(SDHires.as_view(), "/sd/hires")

# CORS settings
cors = CORS(bp, resources={r"/sd/*": {"origins": "*", "headers": "*"}})

# setup sanic app
app = Sanic(__name__)


@bp.after_server_stop
async def close_redis(sanic_app, loop):
    pass
    # if CONFIG['debug_mode']:
    #     await sanic_app.ctx.redis_mq.close()


@app.listener("main_process_start")
async def main_process_start(sanic_app, loop):
    print(f"main_process_start {os.path.abspath('.')}")
    # if CONFIG['debug_mode']:
    #     sanic_app.ctx.redis_mq = RedisMQ(CONFIG['redis']['host'], CONFIG['redis']['port'], CONFIG['redis']['redis_mq'])


class Config:
    RESPONSE_TIMEOUT = 300
    SECRET = "xxxGUIJU_TeCH&^%$"


app.update_config(Config)
app.blueprint(bp)

if __name__ == "__main__":
    if CONFIG['server']['ssl']:
        ssl_config = {"cert": CONFIG['server']['ssl_cert'], "key": CONFIG['server']['ssl_key']}
    else:
        ssl_config = None

    app.run(host=CONFIG['server']['host'],
            port=CONFIG['server']['port'],
            workers=1 if CONFIG['debug_mode'] else CONFIG['server']['workers'],
            access_log=CONFIG['server']['access_log'],
            debug=CONFIG['debug_mode'],
            ssl=ssl_config)

import os

from sanic import Blueprint
from sanic import Sanic
from sanic_cors import CORS

from handlers.main import Login, Logout
from handlers.ocr import Upload, Index, Table, Tts, Main
from lib.common.common_util import mkdir
from lib.redis_mq import RedisMQ
from utils.global_vars import CONFIG

# Blueprint
bp = Blueprint("ocr_ch")
# add_route
bp.static('/static', './static/')
bp.add_route(Upload.as_view(), "/upload")
bp.add_route(Main.as_view(), "/main")
bp.add_route(Index.as_view(), "/")
bp.add_route(Table.as_view(), "/table")
bp.add_route(Tts.as_view(), "/tts")
bp.add_route(Login.as_view(), "/login")
bp.add_route(Logout.as_view(), "/logout")


@bp.after_server_stop
async def close_redis(sanic_app, loop):
    await sanic_app.ctx.redis_mq.close()


@bp.before_server_start
async def init_redis_mq(sanic_app, loop):
    pass
    # await sanic_app.ctx.redis_mq.redis_session.flushdb()


# CORS settings
cors = CORS(bp, resources={r"/upload/*": {"origins": "*", "headers": "*"}})

# setup sanic app
app = Sanic(__name__)


@app.listener("main_process_start")
async def main_process_start(sanic_app, loop):
    print(f"main_process_start {os.path.abspath('.')}")
    mkdir(os.path.join(os.path.abspath(CONFIG['storage_dirpath']['user_dir']), 'wav'))
    mkdir(os.path.join(os.path.abspath(CONFIG['storage_dirpath'].get('user_dir')), 'upload'))
    mkdir(os.path.join(os.path.abspath(CONFIG['storage_dirpath'].get('user_dir')), 'json'))
    sanic_app.ctx.redis_mq = RedisMQ(CONFIG['redis']['host'], CONFIG['redis']['port'], CONFIG['redis']['redis_mq'])


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
            workers=CONFIG['server']['workers'],
            access_log=CONFIG['server']['access_log'],
            debug=CONFIG['debug_mode'],
            ssl=ssl_config)

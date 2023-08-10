import os

from sanic import Blueprint
from sanic import Sanic
from sanic_cors import CORS

from handlers.main import Test
from utils.global_vars import CONFIG

# Blueprint
bp = Blueprint("ai_tasks")
# add_route
bp.add_route(Test.as_view(), "/main")

# CORS settings
cors = CORS(bp, resources={r"/upload/*": {"origins": "*", "headers": "*"}})

# setup sanic app
app = Sanic(__name__)


@app.listener("main_process_start")
async def main_process_start(sanic_app, loop):
    print(f"main_process_start {os.path.abspath('.')}")
    # mkdir(os.path.join(os.path.abspath(CONFIG['storage_dirpath'].get('user_dir')), 'upload'))
    # sanic_app.ctx.redis_mq = RedisMQ(CONFIG['redis']['host'], CONFIG['redis']['port'], CONFIG['redis']['redis_mq'])


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

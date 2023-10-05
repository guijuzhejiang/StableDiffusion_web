import importlib
import logging
import os

from sanic import Blueprint
from sanic import Sanic
from sanic_cors import CORS
# from supabase.lib.client_options import ClientOptions
from wechatpayv3 import WeChatPay, WeChatPayType

from handlers.main import SDGenertae, SDHires, Pay, Query, ImageProvider, QueryPayment, WeChatLogin, FetchUserHistory, UserUpload, QuertDiscount
# from supabase import create_client
from handlers.websocket import sd_genreate
from utils.global_vars import CONFIG

# Blueprint
bp = Blueprint("ai_tasks")
# add_route
bp.add_route(SDGenertae.as_view(), "/sd/generate")
bp.add_route(SDHires.as_view(), "/sd/hires")
bp.add_route(Pay.as_view(), "/wechat/pay")
bp.add_route(Query.as_view(), "/wechat/query")
bp.add_route(QuertDiscount.as_view(), "/discount/query")
bp.add_route(WeChatLogin.as_view(), "/wechat/login")
bp.add_route(QueryPayment.as_view(), "/wechat/query_payment")
bp.add_route(ImageProvider.as_view(), "/user/image/fetch")
bp.add_route(FetchUserHistory.as_view(), "/user/image/history")
bp.add_route(UserUpload.as_view(), "/user/image/upload")
bp.add_websocket_route(sd_genreate, "/sd/io")

# CORS settings
cors = CORS(bp, resources={r"/sd/*": {"origins": "*", "headers": "*"},
                           r"/wechat/*": {"origins": "*", "headers": "*"},
                           r"/user/image/*": {"origins": "*", "headers": "*"},
                           # r"/user/image/history": {"origins": "*", "headers": "*"},
                           })

# setup sanic app
app = Sanic(__name__)
print(app.config.WEBSOCKET_MAX_SIZE)

@bp.after_server_stop
async def close_redis(sanic_app, loop):
    pass
    # if CONFIG['debug_mode']:
    #     await sanic_app.ctx.redis_mq.close()


@app.listener("before_server_start")
async def main_process_start(sanic_app, loop):
    print(f"before_server_start {os.path.abspath('.')}")
    # celery.Celery.control.purge()
    # await celery.connection.connect()

    logging.basicConfig(filename=os.path.join(os.getcwd(), 'demo.log'), level=logging.DEBUG, filemode='a',
                        format='%(asctime)s - %(process)s - %(levelname)s: %(message)s')

    with open(CONFIG['wechatpay']['PRIVATE_KEY'], 'r') as f:
        PRIVATE_KEY = f.read()
        sanic_app.ctx.wxpay = WeChatPay(
            wechatpay_type=WeChatPayType.NATIVE,
            mchid=CONFIG['wechatpay']['MCHID'],
            private_key=PRIVATE_KEY,
            cert_serial_no=CONFIG['wechatpay']['CERT_SERIAL_NO'],
            apiv3_key=CONFIG['wechatpay']['APIV3_KEY'],
            appid=CONFIG['wechatpay']['APPID'],
            notify_url=CONFIG['wechatpay']['NOTIFY_URL'],
            cert_dir=CONFIG['wechatpay']['CERT_DIR'],
            logger=logging.getLogger("wxpay"),
            partner_mode=CONFIG['wechatpay']['PARTNER_MODE'],
            proxy=None)
        # sanic_app.ctx.wxpay.query()
    # supabase_opt = ClientOptions()
    # supabase_opt.postgrest_client_timeout = 20
    # sanic_app.ctx.supabase_client = create_client(CONFIG['supabase']['url'], CONFIG['supabase']['key'],
    #                                               options=supabase_opt)
    sanic_app.ctx.supabase_client = getattr(importlib.import_module('aiosupabase'), 'Supabase')
    sanic_app.ctx.supabase_client.configure(
        url=CONFIG['supabase']['url'],
        key=CONFIG['supabase']['key'],
        debug_enabled=True,
    )


class Config:
    RESPONSE_TIMEOUT = 600
    SECRET = "xxxGUIJU_TeCH&^%$"
    WEBSOCKET_MAX_SIZE = 2097152

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

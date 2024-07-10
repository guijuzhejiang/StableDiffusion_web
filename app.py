import importlib
import logging
import os

from gotrue import AsyncMemoryStorage
from sanic import Blueprint
from sanic import Sanic
# from sanic_cors import CORS

from handlers.login import WeChatLogin, LineLogin, PasswordLogin, GoogleLogin
from handlers.pay import WechatReqPayQRCode, PayPalCreateOrder, WechatQueryPayment, QueryBalance, QueryDiscount, \
    PayPalCaptureOrder, PayPalCreateSub, CheckVip, PayPalCancelSub, PayPalWebhook, ElepayCreateEasyQR, \
    ElepayQueryPayment
from handlers.qinghua import DeviceAuthVerify
from handlers.zs import SDGen, SDBgProvider
from lib.celery_workshop.wokrshop import WorkShop
from operators import OperatorSD, OperatorSora
from wechatpayv3 import WeChatPay, WeChatPayType

from handlers.main import ImageProvider, FetchUserHistory, UserUpload, RevokeTask, SendCaptcha, \
    VerifyCaptcha, UserEditNickname, FetchGallery, FetchVideo
from redis import asyncio as aioredis
from handlers.websocket import sd_genreate, qinghua_genreate
from utils.global_vars import CONFIG
from supabase import acreate_client, ClientOptions

# Blueprint
bp = Blueprint("ai_tasks")
# add_route
bp.add_route(PayPalCreateOrder.as_view(), "/paypal/pay")
bp.add_route(PayPalCaptureOrder.as_view(), "/paypal/query")
bp.add_route(PayPalCreateSub.as_view(), "/paypal/sub")
bp.add_route(PayPalCancelSub.as_view(), "/paypal/cancel")
bp.add_route(PayPalWebhook.as_view(), "/paypal/webhook")

bp.add_route(ElepayCreateEasyQR.as_view(), "/elepay/createqr")
bp.add_route(ElepayQueryPayment.as_view(), "/elepay/query_payment")
bp.add_route(CheckVip.as_view(), "/user/check")

bp.add_route(WechatReqPayQRCode.as_view(), "/wechat/pay")
bp.add_route(QueryBalance.as_view(), "/wechat/query")
bp.add_route(QueryDiscount.as_view(), "/discount/query")
bp.add_route(RevokeTask.as_view(), "/management/revoke_task")
bp.add_route(SendCaptcha.as_view(), "/sms/send_captcha")
bp.add_route(VerifyCaptcha.as_view(), "/sms/verify_captcha")
bp.add_route(WeChatLogin.as_view(), "/wechat/login")
bp.add_route(LineLogin.as_view(), "/line/login")
bp.add_route(PasswordLogin.as_view(), "/user/login")
bp.add_route(GoogleLogin.as_view(), "/google/login")
# bp.add_route(LineLoginPost.as_view(), "/line/login_post")
bp.add_route(WechatQueryPayment.as_view(), "/wechat/query_payment")
bp.add_route(ImageProvider.as_view(), "/user/image/fetch")
bp.add_route(FetchUserHistory.as_view(), "/user/image/history")
bp.add_route(FetchGallery.as_view(), "/gallery/fetch")
bp.add_route(UserUpload.as_view(), "/user/image/upload")
bp.add_route(UserEditNickname.as_view(), "/user/edit/nickname")
bp.add_route(FetchVideo.as_view(), "/user/video/fetch")

bp.add_route(DeviceAuthVerify.as_view(), "/qinghua/device/auth")
bp.add_websocket_route(qinghua_genreate, "/qinghua/device/generate")

bp.add_route(SDGen.as_view(), "/zs/bg/generate")
bp.add_route(SDBgProvider.as_view(), "/learninglang/image/fetch")
bp.add_websocket_route(sd_genreate, "/sd/io")

# CORS settings
# cors = CORS(bp, resources={
    # r"/sd/*": {"origins": "*", "headers": "*"},
    #                        r"/wechat/*": {"origins": "*", "headers": "*"},
    #                        r"/user/image/upload": {"origins": "*", "headers": "*"},
                           # r"/user/image/history": {"origins": "*", "headers": "*"},
                           # })

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
    supabase_opt = ClientOptions(storage=AsyncMemoryStorage(), postgrest_client_timeout=20)
    # supabase_opt.postgrest_client_timeout = 20
    sanic_app.ctx.supabase_client = await acreate_client(CONFIG['supabase']['url'], CONFIG['supabase']['key'],
                                                         options=supabase_opt)
    # sanic_app.ctx.supabase_client = getattr(importlib.import_module('aiosupabase'), 'Supabase')
    # sanic_app.ctx.supabase_client.configure(
    #     url=CONFIG['supabase']['url'],
    #     key=CONFIG['supabase']['key'],
    #     debug_enabled=False,
    # )
    # sanic_app.ctx.supabase_client = await create_client(CONFIG['supabase']['url'], CONFIG['supabase']['key'])

    sanic_app.ctx.redis_session = aioredis.from_url(f"redis://localhost:6379/1", decode_responses=True)
    sanic_app.ctx.redis_session_sms = aioredis.from_url(f"redis://localhost:6379/2", decode_responses=True)
    await sanic_app.ctx.redis_session_sms.flushdb()

    sanic_app.ctx.sd_workshop = WorkShop(OperatorSD)
    sanic_app.ctx.sora_workshop = WorkShop(OperatorSora)

    os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID'] = CONFIG['aliyun']['sms']['access_key_id']
    os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET'] = CONFIG['aliyun']['sms']['access_key_secret']


class Config:
    RESPONSE_TIMEOUT = 600
    SECRET = "xxxGUIJU_TeCH&^%$"
    WEBSOCKET_MAX_SIZE = 2097152
    USE_UVLOOP = False if CONFIG['debug_mode'] else True

app.update_config(Config)
app.blueprint(bp)


if __name__ == "__main__":
    if CONFIG['server']['ssl']:
        ssl_config = {"cert": CONFIG['server']['ssl_cert'], "key": CONFIG['server']['ssl_key']}
    else:
        ssl_config = None

    app.run(host=CONFIG['server']['host'],
            port=CONFIG['server']['port'],
            workers=1,
            access_log=CONFIG['server']['access_log'],
            debug=CONFIG['debug_mode'],
            ssl=ssl_config)

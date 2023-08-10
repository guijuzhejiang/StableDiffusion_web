import json
import os
import traceback

from lib.common.common_util import logging
from lib.redis_mq import SyncRedisMQ
from lib.redis_pipline.pipeline import Pipeline
from utils.global_vars import CONFIG

if __name__ == '__main__':
    try:
        from operators.ocr_preprocess import OperatorPre
        from operators.ocr_process import OperatorOCR
        from operators.text_to_speech import OperatorTextToSpeech

        """ init RedisMQ """
        redis_mq = SyncRedisMQ(CONFIG['redis']['host'], CONFIG['redis']['port'], CONFIG['redis']['redis_mq'], 'clothing_inpaint')
        """ define work line """
        workline = [OperatorPre, OperatorOCR]

        if CONFIG['debug_mode']:
            os.makedirs('data/debug', exist_ok=True)

            debug_list = []
            for w in workline:
                debug_list.append(w())

        else:
            pipeline = Pipeline(workline)
            print("pipeline running...")

        # start consuming
        for msg_id, msg in redis_mq.consume():
            logging(
                f"{__file__}got msg: {msg}",
                f"logs/info.log", print_msg=CONFIG['debug_mode'])
            json_msg = json.loads(msg['json_msg'])
            procedure = [OperatorPre.consume_queue_name, OperatorOCR.consume_queue_name]
            msg['pipeline'] = ','.join(procedure)

            # del json_msg['operation']
            # msg['json_msg'] = json.dumps(json_msg)
            if CONFIG['debug_mode']:
                if json_msg['operation'] == 'ocr':
                    for i in range(2):
                        msg = debug_list[i].operation(msg)
                        if 'error' in msg.keys():
                            break
                else:
                    msg = debug_list[-1].operation(msg)

                redis_mq.pub(msg['reply_queue_name'], msg, CONFIG['server']['msg_expire_secs'])
            else:
                if json_msg['operation'] == 'ocr':
                    redis_mq.pub(procedure[0], msg)
                else:
                    redis_mq.pub(OperatorTextToSpeech.consume_queue_name, msg)
    except Exception:
        print(traceback.format_exc())
        logging(
            f"{__file__} fatal error: {traceback.format_exc()}",
            f"logs/error.log", print_msg=CONFIG['debug_mode'])

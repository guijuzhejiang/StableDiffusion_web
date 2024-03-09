# coding=utf-8
# @Time : 2024/1/30 下午5:26
# @File : sub.py
import asyncio
import base64
import datetime
import os
import sys
import traceback

import requests
import yaml
from crontab import CronTab

proj_dir_path = os.path.dirname(os.path.dirname(__file__))
CONFIG = yaml.safe_load(open(os.path.join(proj_dir_path, "config.yml"), 'r'))


def logging(msg, print_msg=False):
    msg = f"{str(datetime.datetime.now())} : {msg}"
    if print_msg:
        print(msg)

    with open(os.path.join(proj_dir_path, 'logs', 'cron.log'), 'a') as f:
        f.write(msg + "\n")


def aspayapl_generate_ccess_token():
    # Encode client ID and client secret as base64
    auth = base64.b64encode(f"{CONFIG['paypal']['client_id']}:{CONFIG['paypal']['client_secret']}".encode()).decode()

    response = requests.post(f"{CONFIG['paypal']['base_url']}/v1/oauth2/token",
                             data="grant_type=client_credentials",
                             headers={"Authorization": f"Basic {auth}"})
    # Parse the JSON response
    data = response.json()

    # Return the access token
    return data["access_token"]


def setup_cron(sub_id, target_date):
    cron = CronTab(user=True)

    # 添加一个 cron 作业
    job = cron.new(command=f'{sys.executable} {__file__} {sub_id}')

    # 设置作业执行的时间，这里设置为每个月的1号10点30分
    job.minute.on(target_date.minute)
    job.hour.on(target_date.hour)
    job.day.on(target_date.day)
    job.month.on(target_date.month)
    job.year.on(target_date.year)
    job.set_comment(f'{sub_id}')

    # 将 cron 作业写入到 crontab 文件中
    cron.write()


async def run_main():
    try:
        from aiosupabase import Supabase

        subscription_id = sys.argv[1]
        supabase_client = Supabase
        supabase_client.configure(
            url=CONFIG['supabase']['url'],
            key=CONFIG['supabase']['key'],
            debug_enabled=True,
        )
        # supabase_client = await create_client(CONFIG['supabase']['url'], CONFIG['supabase']['key'])
        subscription_data = (await supabase_client.atable("subscription").select("*").eq('subscription_id',
                                                                                 subscription_id).execute()).data
        user_id = subscription_data[0]['user_id']

        if len(subscription_data) > 0:
            # show sub detail
            access_token = aspayapl_generate_ccess_token()
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            }

            if CONFIG['local']:
                response = requests.get(f'https://api-m.sandbox.paypal.com/v1/billing/subscriptions/{subscription_id}',
                                        headers=headers)
            else:
                response = requests.get(f'https://sandbox.paypal.com/v1/billing/subscriptions/{subscription_id}',
                                        headers=headers)

            account = (await supabase_client.atable("account").select("*").eq("id", user_id).execute()).data[0]
            vip_level = account['vip_level'][0]

            if vip_level == 1:
                add_balance = 220
            elif vip_level == 2:
                add_balance = 720
            # else:
            #     add_balance = 624

            if response.json()['status'] == 'ACTIVE':
                await supabase_client.atable("account").update({'balance': account['balance'] + add_balance}).execute()
                data = await supabase_client.atable("transaction").insert({"user_id": user_id,
                                                                     'out_trade_no': f'@subpaypal_{subscription_id}',
                                                                     'amount': add_balance,
                                                                     'is_plus': True,
                                                                     'status': 1,
                                                                     }).execute()

            else:
                await supabase_client.atable("account").update({'vip_level': 0}).execute()
                data = await supabase_client.atable("subscription").delete().eq("user_id", user_id).execute()

    except Exception:
        logging(traceback.format_exc())

    finally:
        logging(f'try to clean {subscription_id}')
        cron = CronTab(user=True)
        cron.remove_all(comment=f'{subscription_id}')
        logging(f'clean {subscription_id} success')


if __name__ == '__main__':
    asyncio.run(run_main())

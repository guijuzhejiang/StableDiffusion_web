sudo supervisorctl start|stop all|sd_factory|sd_webbackend

tail -f factory.info.log

[program:sd_webbackend]
command=/home/zzg/miniconda3/envs/py310_sd_web/bin/python app.py ;
environment=PYTHONPATH=/home/zzg/miniconda3/envs/py310_sd_web:/home/zzg/workspace/pycharm/StableDiffusion_web:/home/zzg/miniconda3/envs/py310_sd_web/lib/python3.10:/home/zzg/miniconda3/envs/py310_sd_web/lib/python3.10/lib-dynload:/home/zzg/miniconda3/envs/py310_sd_web/lib/python3.10/site-packages;
directory=/home/zzg/workspace/pycharm/StableDiffusion_web ;
autorestart=true ;
stderr_logfile=/home/zzg/workspace/pycharm/StableDiffusion_web/app.err.log ;
stdout_logfile=/home/zzg/workspace/pycharm/StableDiffusion_web/app.info.log ;
user=zzg ;
autostart=start ;
stopsignal=STOP ;


[program:sd_factory]
command=/home/zzg/miniconda3/envs/py310_sd_service/bin/python async_run_ai_factory.py ;
environment=PYTHONPATH=/home/zzg/miniconda3/envs/py310_sd_service:/home/zzg/workspace/pycharm/StableDiffusion_web:/home/zzg/miniconda3/envs/py310_sd_service/lib/python3.10:/home/zzg/miniconda3/envs/py310_sd_service/lib/python3.10/lib-dynload:/home/zzg/miniconda3/envs/py310_sd_service/lib/python3.10/site-packages;
directory=/home/zzg/workspace/pycharm/StableDiffusion_web ;
autorestart=true ;
stderr_logfile=/home/zzg/workspace/pycharm/StableDiffusion_web/factory.err.log ;
stdout_logfile=/home/zzg/workspace/pycharm/StableDiffusion_web/factory.info.log ;
user=zzg ;
autostart=true ;
stopsignal=STOP ;
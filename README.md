sudo supervisorctl start|stop all|sd_factory|sd_webbackend

tail -f factory.info.log

[program:sd_webbackend]
command=/home/zzg/miniconda3/envs/py310_sd_web/bin/python app.py ;
environment=PYTHONPATH=/home/zzg/miniconda3/envs/py310_sd_web:/home/zzg/workspace/pycharm/StableDiffusion_web:/home/zzg/miniconda3/envs/py310_sd_web/lib/python3.10:/home/zzg/miniconda3/envs/py310_sd_web/lib/python3.10/lib-dynload:/home/zzg/miniconda3/envs/py310_sd_web/lib/python3.10/site-packages ;
directory=/home/zzg/workspace/pycharm/StableDiffusion_web ;
autorestart=true ;
stderr_logfile=/home/zzg/workspace/pycharm/StableDiffusion_web/app.err.log ;
stdout_logfile=/home/zzg/workspace/pycharm/StableDiffusion_web/app.info.log ;
user=zzg ;
autostart=true ;
stopsignal=STOP ;


[program:sd_factory]
command=/home/zzg/miniconda3/envs/py310_sd_service/bin/python async_run_ai_factory.py ;
environment=PYTHONPATH="/home/zzg/miniconda3/envs/py310_sd_service:/home/zzg/workspace/pycharm/StableDiffusion_web:/home/zzg/miniconda3/envs/py310_sd_service/lib/python3.10:/home/zzg/miniconda3/envs/py310_sd_service/lib/python3.10/lib-dynload:/home/zzg/miniconda3/envs/py310_sd_service/lib/python3.10/site-packages",PATH="/home/zzg/.poetry/bin:/home/zzg/.local/bin:/home/zzg/.local/share/pnpm:/home/zzg/.nvm/versions/node/v18.17.1/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/home/zzg/pkg/neo4j-community-4.3.7/bin:/usr/lib/jvm/java-11-openjdk-amd64/bin:/home/zzg/miniconda3/envs/py310_sd_service/bin:/home/zzg/miniconda3/condabin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/llvm11/bin:/home/zzg/pkg/TensorRT-7.2.3.4/bin:/home/zzg/.cargo/bin",LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib:/usr/local/cuda/lib64:/usr/local/lib::/usr/local/TensorRT-8.6.1.6/lib";
directory=/home/zzg/workspace/pycharm/StableDiffusion_web ;
autorestart=true ;
stderr_logfile=/home/zzg/workspace/pycharm/StableDiffusion_web/factory.err.log ;
stdout_logfile=/home/zzg/workspace/pycharm/StableDiffusion_web/factory.info.log ;
user=zzg ;
autostart=true ;
stopsignal=QUIT ;
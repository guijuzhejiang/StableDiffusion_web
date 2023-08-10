from collections import OrderedDict

from lib.common.NoDaemonProcessPool import NoDaemonProcess
from lib.redis_pipline.op_proc import WorkerProc


class Pipeline(object):
    op_list = []
    qn_list = []
    proc_list = []
    redis_mq = None
    temp = None

    def __init__(self, op_list):
        self.op_list = op_list
        # self.op_list[0].first_op = True
        # self.op_list[-1].last_op = True
        print(self.op_list)
        for index, op in enumerate(self.op_list):
            self.qn_list.append(f"{op.__name__}")
            args = OrderedDict({'worker_num': op.num,
                                'op': op.__name__,
                                'consume_queue_name': op.consume_queue_name,
                                })

            # t_proc = subprocess.Popen(
            #     [sys.executable, os.path.abspath('lib/redis_pipline/op_proc.py')] + [str(i) for i in args.values()],
            #     shell=False,
            #     stdin=subprocess.PIPE,
            #     stdout=subprocess.PIPE,
            #     stderr=subprocess.PIPE)
            # self.proc_list.append(t_proc)

            # t_proc = subprocess.Popen(
            #     f"{sys.executable} {os.path.abspath('lib/redis_pipline/op_proc.py')} {' '.join([str(i) for i in args.values()])}",
            #     shell=True,
            #     stdin=subprocess.PIPE,
            #     stdout=subprocess.PIPE,
            #     stderr=subprocess.PIPE)
            # self.proc_list.append(t_proc)

            self.proc_list.append(NoDaemonProcess(target=WorkerProc.worker_main_proc, args=(str(i) for i in args.values()))),

        [p.start() for p in self.proc_list]

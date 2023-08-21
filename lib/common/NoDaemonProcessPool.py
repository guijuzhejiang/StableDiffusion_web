from multiprocessing import get_context, pool


class NoDaemonProcess(get_context('spawn').Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

# def get_no_daemon_process(target, args, method='spawn'):
#     class NoDaemonProcess(get_context(method).Process):
#         @property
#         def daemon(self):
#             return False
#
#         @daemon.setter
#         def daemon(self, value):
#             pass
#
#     return NoDaemonProcess(target=target, args=args)
# class NoDaemonContext(type(get_context())):
#     Process = NoDaemonProcess
#
#
# # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# # because the latter is only a wrapper function, not a proper class.
# class ProcessPool(pool.Pool):
#     def __init__(self, *args, **kwargs):
#         kwargs['context'] = NoDaemonContext()
#         super(ProcessPool, self).__init__(*args, **kwargs)


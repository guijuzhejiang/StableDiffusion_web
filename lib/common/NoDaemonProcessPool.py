from multiprocessing import get_context, pool


# class NoDaemonProcess(multiprocessing.Process):
#     # make 'daemon' attribute always return False
#     def _get_daemon(self):
#         return False
# 
#     def _set_daemon(self, value):
#         pass
# 
#     daemon = property(_get_daemon, _set_daemon)
# 
# 
# # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# # because the latter is only a wrapper function, not a proper class.
# class ProcessPool(multiprocessing.pool.Pool):
# 
#     Process = NoDaemonProcess
ctx = get_context('fork')


class NoDaemonProcess(ctx.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(get_context())):
    Process = NoDaemonProcess


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class ProcessPool(pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(ProcessPool, self).__init__(*args, **kwargs)


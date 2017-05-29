class Observable(object):
    def register(self, observer):
        pass

    def deregister(self, observer):
        pass

    def deregister_all(self):
        pass

    def notify_all(self, *args, **kwargs):
        pass
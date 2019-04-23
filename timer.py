from datetime import datetime


_time = None

def start():
    global _time
    _time = datetime.now()

def end():
    print('Time: ' + str(datetime.now() - _time))

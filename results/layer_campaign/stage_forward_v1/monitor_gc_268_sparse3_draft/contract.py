"""MONITOR-only ABBA268 contract; same real-SGD replay and fullwall estimand."""
ORDER='ABBA';WARMUP=12;LENGTH=64;TOTAL=268
def position(physical):
    if type(physical) is not int or not 1<=physical<=TOTAL:raise ValueError('physical iteration outside1..268')
    if physical<=WARMUP:return dict(physical=physical,block=0,mode='A',logical=physical,measured=False,profile=False)
    block,offset=divmod(physical-WARMUP-1,LENGTH);logical=offset+1
    return dict(physical=physical,block=block+1,mode=ORDER[block],logical=logical,measured=logical>=33,profile=False)

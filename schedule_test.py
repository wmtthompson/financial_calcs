import schedule
import time
import os

def hello():
    print("Hello World!")
    duration = 1
    freq = 440
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

schedule.every(1).minutes.do(hello)

while True:
    schedule.run_pending()
    time.sleep(2)

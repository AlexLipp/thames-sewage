import time as time
from datetime import datetime
import sewage as swg


def make_maps_continuously(repeat_time: float = 900):
    """Repeatedly calls `make_discharge_map` at specified repeat time
    given in seconds. Default repeat time is 15 minutes (900 seconds).
    Note, if execution time exceeds the run"""
    while True:
        start = time.time()
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Starting @", now)
        swg.make_discharge_map()
        elapsed = time.time() - start
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print("Finished calculation at", now)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        if elapsed > repeat_time:
            print("!!! Warning !!!\nRuntime exceeded repeat time at " + now)
            print("Consider adjusting repeat-time or decreasing runtime.")
            print("Continuing...")
        else:
            print("... pausing for ~ ", repeat_time / 60, "minutes ...")
            time.sleep(repeat_time - elapsed)


def main():
    make_maps_continuously(1)


if __name__ == "__main__":
    main()

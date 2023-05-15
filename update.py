import sewage as swg
from datetime import datetime


def main():
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Starting @", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print("Retrieving discharges to date")
    swg.get_all_past_discharges()
    print("Calculating downstream discharge information")
    swg.make_discharge_map()
    print("Finished @", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


if __name__ == "__main__":
    main()

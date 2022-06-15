import os

from funcs.dir import d_drive
from funcs.post_processing.tube.process import process_multiple_days

FOIL_DAYS = (
    "2020-11-12",
    "2020-11-13",
    "2020-11-23",
    "2020-11-24",
    "2020-11-25",
    "2020-11-26",
    "2020-12-07",
    "2020-12-08",
    "2020-12-09",
    "2020-12-10",
    "2020-12-20",
    "2020-12-21",
    "2020-12-27",
)


def main():
    loc_output = os.path.join(d_drive, "Data", "Processed", "Soot Foil", "tube_data.h5")
    process_multiple_days(
        dates_to_process=FOIL_DAYS,
        loc_processed_h5=loc_output,
        overwrite=True,
    )


if __name__ == "__main__":
    main()

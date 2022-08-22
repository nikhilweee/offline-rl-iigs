import os
import logging
import leduc_cfr_collect
import leduc_cfr_offline

logging.basicConfig(
    format=('%(asctime)s.%(msecs)03d %(levelname)s '
            '%(module)s L%(lineno)03d | %(message)s'),
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__file__)

if __name__ == '__main__':
    for coverage in [500, 600, 700, 800, 900, 936]:

        logger.info(f"Collecting data with coverage {coverage}")

        args = leduc_cfr_collect.parse_args()
        args.coverage = coverage
        args.label = f'collect'
        info_states = leduc_cfr_collect.main(args)

        logger.info(f"Running Offline CFR with coverage {info_states}")

        pkl_file = f'trajectories/traj-mixed-{info_states}.pkl'
        args = leduc_cfr_offline.parse_args()
        args.traj = pkl_file
        args.label = f'offline-{info_states}'
        leduc_cfr_offline.main(args)


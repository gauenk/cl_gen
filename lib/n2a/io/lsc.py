#!/user/bin/python3.8

import sys


if len(sys.argv) == 1:
    print("lsc allows users to quickly read cache info.")
    exit()
path = sys.argv[1]
exp_id = exp_id_from_path(exp_id)
exp_info = get_exp_info_from_exp_id(exp_id)
print_formatted_exp_info(exp_info)

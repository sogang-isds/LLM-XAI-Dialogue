import json
import linecache
import logging
import os
import sys


def save_to_json(data, filename='data.json'):
    if filename[-4:] != 'json':
        filename += '.json'

    with open(f'{filename}', 'w', encoding='utf-8') as fw:
        json.dump(data, fw, indent=4, ensure_ascii=False)


def load_json(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def read_file(file):
    lines = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())

    return lines


def write_list_to_file(lines, filename='data.txt'):
    with open(filename, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f'{line}\n')


def exit_program(msg="SystemExit : Terminate program."):
    sys.exit(msg)


def create_symbolic_links(path_dict):
    for source_dir, source_path in path_dict.items():
        try:
            os.remove(f'./{source_dir}')
        except FileNotFoundError:
            pass
        finally:
            if os.path.exists(source_path):
                os.symlink(source_path, f'./{source_dir}')
            else:
                logging.error(f"Can't import source from {source_path}")
                exit_program()


def print_exception(use_logger=False, exit=False):
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    line = line.strip()

    msg = 'EXCEPTION IN ({}, LINE: {}, CODE: {}): {} {}'.format(filename, lineno, line, exc_type, exc_obj)

    if use_logger:
        logging.error(msg)
    else:
        print(msg)

    if exit:
        sys.exit(1)

    return msg

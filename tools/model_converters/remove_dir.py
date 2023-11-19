import os
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description='Remove files')
    parser.add_argument('dir_root', help='input dir path')
    parser.add_argument('rm_name', type=str, help='filename to remove')
    args = parser.parse_args()
    return args


def process_remove(dir_root, rm_name=None):
    assert rm_name is not None
    assert os.path.exists(dir_root)
    dir_list = os.listdir(path=dir_root)
    if len(rm_name.split('-')) > 1:
        rm_name = rm_name.split('-')
    else:
        rm_name = [rm_name]

    for dir in dir_list:
        for _name in rm_name:
            _path = os.path.join(dir_root, dir, _name)
            try:
                if '*' in _path:
                    _path_base, suffix = _path.split('*')
                    _path_dir = os.listdir(path=_path_base)
                    for _path_name in _path_dir:
                        if _path_name.startswith(suffix) or _path_name.endswith(suffix):
                            _path_rm = os.path.join(_path_base, _path_name)
                            if os.path.isfile(_path_rm):
                                subprocess.Popen(['rm', _path_rm])
                            else:
                                print('fail to remove:', _path_rm)

                elif os.path.isfile(_path):
                    subprocess.Popen(['rm', _path])
                else:
                    subprocess.Popen(['rm', '-r', _path])
            except:
                print('fail to remove:', _path)


def main():
    args = parse_args()
    process_remove(args.dir_root, args.rm_name)


if __name__ == '__main__':
    main()

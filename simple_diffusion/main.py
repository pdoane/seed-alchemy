import os

os.environ['DISABLE_TELEMETRY'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] ='1'

import argparse
import sys

from . import configuration
from .application import Application


def main():
    if sys.platform == 'darwin':
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        info_dict = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        info_dict['CFBundleName'] = configuration.APP_NAME

    parser = argparse.ArgumentParser(description=configuration.APP_NAME)
    parser.add_argument('--root')
    args = parser.parse_args()

    configuration.set_resources_path(os.path.join(os.getcwd(), 'simple_diffusion/resources'))
    if args.root:
        os.chdir(os.path.expanduser(args.root))

    app = Application(sys.argv)
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

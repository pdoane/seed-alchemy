import os

os.environ['DISABLE_TELEMETRY'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] ='1'

import sys

import configuration
from application import Application

def main():
    if sys.platform == 'darwin':
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        info_dict = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        info_dict['CFBundleName'] = configuration.APP_NAME

    app = Application(sys.argv)
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

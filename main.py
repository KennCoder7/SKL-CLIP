#!/usr/bin/env python
import argparse
import sys

# torchlight
import torchlight
from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    processors['recognition_clip'] = import_class('processor.recognition_clip.REC_Processor')
    processors['recognition_clip_parallel'] = import_class('processor.recognition_clip_parallel.REC_Processor')
    processors['recognition_clip_parallel_UDA'] = import_class('processor.recognition_clip_parallel_UDA.REC_Processor')
    processors['recognition_clip_parallel_ZSL'] = import_class('processor.recognition_clip_parallel_ZSL.REC_Processor')
    processors['recognition_baseline'] = import_class('processor.recognition_baseline.REC_Processor')



    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    p.start()

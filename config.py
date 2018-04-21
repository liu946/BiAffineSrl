#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Config usage:
#   import first
#   add_options in the global part
#   get_options in the __init__ function
#   don't forget to call parse_args() in the main file.
#
import optparse

__all__ = ['config_add_option', 'config_get_option', 'parse_args']

config_parser = optparse.OptionParser(usage='Usage:%prog [option]')
config = {}

def add_option(*args, **kwargs):
    config_parser.add_option(*args, **kwargs)

def get_option(key):
    return config.__getattribute__(key)

def parse_args():
    global config
    config, _ = config_parser.parse_args()

add_option('--save', dest='save', default='saves/default', type='string', help='path to save files', action='store')

if __name__ == '__main__':
    config_parser.add_option('-t','--test',dest='test_variable',default='default',type='string',help='It is a test option',action='store')
    config_parser.parse_args()
    print(config_parser.get_option('test_variable'))

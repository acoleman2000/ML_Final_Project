__version__ = "0.0.0"
__status__ = "Development"

import sys
import argparse
import random
import os, os.path
from create_signals import CadenceGroup
def usr_args():
    """
    functional arguments for process
    https://stackoverflow.com/questions/27529610/call-function-based-on-argparse
    """

    # initialize parser
    parser = argparse.ArgumentParser()

    # set usages options
    parser = argparse.ArgumentParser(
        prog='insert_signals',
        usage='%(prog)s [options]')

    # version
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='%(prog)s ' + __version__)

    # create subparser objects
    subparsers = parser.add_subparsers()

    # Create parent subparser. Note `add_help=False` & creation via `argparse.`
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('-o', '--output',
                               required=False,
                               help="Output directory to store files in")

    parent_parser.add_argument('-l', '--log',
                                required=False,
                                help="Log of all files generated")

    parent_parser.add_argument('-i', '--interesting',
                                nargs='?',
                                const=0,
                                type=int,
                                default=0)


    parent_parser.add_argument('-u', '--uninteresting',
                                nargs='?',
                                const=0,
                                type=int,
                                default=0)




    # Create a functions subcommand
    parser_listapps = subparsers.add_parser('functions',
                                            help='List all available functions.')
    parser_listapps.set_defaults(func=list_apps)

    # Create the bco_license
    parser_generate_images = subparsers.add_parser('generate_images',
                                           parents=[parent_parser],
                                           help='Generate waterfall images with signals.')
    parser_generate_images.set_defaults(func=generate_images)

    # Create a validate subcommand
    parser_generate_h5 = subparsers.add_parser('generate_h5',
                                            parents=[parent_parser],
                                            help="generate .h5 files with signals")
    parser_generate_h5.set_defaults(func=generate_h5)




    # Print usage message if no args are supplied.
    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    # Run the appropriate function
    options = parser.parse_args()
    if options.func is list_apps:
        options.func(parser)
    else:
        options.func(options)

def list_apps(parser: argparse.ArgumentParser):
    """
    List all functions and options available in app
    https://stackoverflow.com/questions/7498595/python-argparse-add-argument-to-multiple-subparsers
    """

    print('Function List')
    subparsers_actions = [
        # pylint: disable=protected-access
        action for action in parser._actions
        # pylint: disable=W0212
        if isinstance(action, argparse._SubParsersAction)]
    # there will probably only be one subparser_action,
    # but better safe than sorry
    for subparsers_action in subparsers_actions:
        # get all subparsers and print help
        for choice, subparser in subparsers_action.choices.items():
            print("Function: '{}'".format(choice))
            print(subparser.format_help())
    # print(parser.format_help())



def generate_images(options: dict):
    log_file = "manifests/manifest_%s.csv"%(len([name for name in os.listdir('manifests') if os.path.isfile("manifests/%s"%name)]) + 1)
    with open(log_file, "w+") as f:
        f.write("image,#classification\n")
    for i in range(0, options.uninteresting):
       num_rfi = random.randint(1,4)
       cadenceGroup = CadenceGroup(save_dir = options.output + "/uninteresting/", name=str(i), num_rfi=num_rfi)
       cadenceGroup.update_frames()
       cadenceGroup.save_waterfall_plots(log_file)
       cadenceGroup.save_h5_files()

    for i in range(0, options.interesting):
        num_rfi = random.randint(0,3)
        cadenceGroup = CadenceGroup(save_dir = options.output + "/interesting/", name=str(i), num_rfi=num_rfi, num_interesting=1)
        cadenceGroup.update_frames()
        cadenceGroup.save_waterfall_plots(log_file)
        cadenceGroup.save_h5_files()





def generate_h5(options:dict):

    for i in range(0, options.uninteresting):
       cadenceGroup = CadenceGroup(save_dir = options.output + "/uninteresting/", name=str(i), num_rfi=random.randint(1,3),fchans=random.randint(10000,100000))
       cadenceGroup.update_frames()
       cadenceGroup.save_h5_files()

    for i in range(0, options.interesting):
        cadenceGroup = CadenceGroup(save_dir = options.output + "/interesting/", name=str(i), num_rfi=random.randint(1,3), num_interesting=random.randit(1,3),fchans=random.randint(10000,100000))
        cadenceGroup.update_frames()
        cadenceGroup.save_h5_files()

if __name__ == "__main__":
    usr_args()
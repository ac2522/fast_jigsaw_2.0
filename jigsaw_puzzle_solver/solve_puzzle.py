""" solve_puzzle.py
This script reads all puzzle pieces with filenames starting with $in_prefix
and reconstructs them back to the original image.

e.g)
python solve_puzzle.py test_fragments
"""

import os
import sys
import time
from argparse import ArgumentParser
from configparser import ConfigParser
import cProfile

from solver import assembler as asm

DEFAULT_CONFIG = {
    "config": {
        "pieces_dir": "jigsaw_pieces",
        "output_dir": "images_out",
        "debug": False,
        "show_assembly_animation": True,
        "animation_interval_millis": 200
    }
}


def main(jigsaw_pieces_dir="image_fragments", out_dir="images_out", verbose=False,
         show_anim=True, anim_interval=200, show_mst_on_anim=False):
    """
    main jigsaw puzzle solver routine.
    """
    s_time = time.time()
    # initialize
    assembler = asm.ImageAssembler.load_from_filepath(jigsaw_pieces_dir, verbose, out_dir)
    print("assemble_image.py:", len(assembler.raw_imgs), "files loaded", flush=True)
    if not verbose:
        sys.stdout = open(os.devnull, 'w')  # block stdout

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # main merge algorithm
    try:
        assembler.assemble()
    except (KeyError, TypeError):
        print("Failed intial search, attemtping agian with 50% accuracy")
        try:
            assembler.reset()
            assembler.assemble()
        except (KeyError, TypeError):
            print("Failed secondary search, attemtping agian with 100% accuracy")
            assembler.reset(True)
            assembler.assemble()

    # save result to output directory, and show animation.
    assembler.save_assembled_image(out_dir + "/finished")
    sys.stdout = sys.__stdout__  # restore stdout
    print("total elapsed time:", time.time() - s_time, "seconds", flush=True)
    if show_anim:
        assembler.start_assembly_animation(show_mst_on_anim, anim_interval)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--verbose', '-v', required=False, action='store_true',
                    help='increase output verbosity')
    ap.add_argument('--show_animation', '-a', required=False, action='store_true',
                    help='show image reconstruction animation')
    ap.add_argument('--show_spanning_tree', '-t', required=False, action='store_true',
                    help='show minimum spanning tree on top of the animation (-a option requried)')
    ap.add_argument('--config_file', '-c', required=False, default="./config/config.ini",
                    action='store', nargs=1, help='configuration ini file')
    args = ap.parse_args()

    cp = ConfigParser()
    cp.read_dict(DEFAULT_CONFIG)
    cp.read(args.config_file)
    main(cp.get("config", "jigsaw_pieces_dir"), cp.get("config", "output_dir"),
         cp.getboolean("config", "debug") or args.verbose,
         cp.getboolean("config", "show_assembly_animation") or args.show_animation,
         int(cp.get("config", "animation_interval_millis")), args.show_spanning_tree
    )

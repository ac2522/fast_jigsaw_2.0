""" assembler.py
main jigsaw puzzle solver algorithm.

This script reads all image pieces with provided prefix and reconstructs pieces back to the original images.
It adapts Prim's Minimum Spanning Tree algorithm, and computes a 3D distance matrix in parallel.
"""
import os
import time
#import pickle as pkl
#from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd


from math import log2

from . import assembler_helpers as helpers
from . import assembler_visualizer as vis
from .assembler_data_structures import Image, Config, PuzzleBlock, LinkedHashmapPriorityQueue


# The threshold for the number of images required to trigger parallel distance matrix computation.
PARALLEL_COMPUTATION_MIN_IMAGES = 96
# The number of CPU cores to be used to construct the similarity matrix in parallel
PROCESS_COUNT = 8

TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3
HOLE, KNOB, FLAT = 0, 1, 2


class ImageAssembler:
    """
    Jigsaw puzzle assembler class.

    Usage:
        from jigsaw_puzzle_solver.assembler import imageAssembler
        assembler = ImageAssembler.load_from_filepath(directory, prefix, max_cols, max_rows)
        assembler.assemble()
        assembler.start_assembly_animation()
        assembler.save_assembled_image()

    Attributes:
        raw_imgs (2d list of cv2 images):
            collection of puzzle pieces with all possible orientations
            (4 for square pieces, 8 for rectangle ones)

    Methods:
        assemble() -> void
        save_assembled_image() -> void
        save_assembled_image(filepath: str) -> void
        start_assembly_animation(interval_millis: int) -> void
    """

    def __init__(self, raw_imgs=[], verbose=False, out_dir=None):
        self.raw_imgs = raw_imgs  # rectangular images with all orientations, aligned to match width & height.
        self.img_cnt = len(self.raw_imgs)
        self.pieces_placed = 0
        self.edge_cnt = 4 * self.img_cnt
        im_Start = raw_imgs[0]
        self.img_size = im_Start.shape[0]

        overlap = np.argmax(im_Start[self.img_size//2, :self.img_size//2, 3])
        self.piece_sz = self.img_size - 2 * overlap
        self.accuracy = max(int(log2(self.piece_sz) * 4 - 10), 10)
        self.edge_maps = [{}, {}]
        self.merge_history = []

        self.blueprint = None  # blueprint for image reconstruction
        self.whiteboard = None
        Config.half = self.img_size // 2
        Config.overlap = overlap
        Config.samples = np.linspace(overlap, overlap + self.piece_sz, self.accuracy + 4, dtype=np.int32)[2:-2]
        Config.samples_reversed = self.img_size - 1 - Config.samples
        Config.IDX_S = self.accuracy // 6
        Config.IDX_K = (self.img_size + 3 * overlap) // 5
        self.verbose, self.out_dir = verbose, out_dir 


    @classmethod
    def load_from_filepath(cls, directory, verbose=False, out_dir=None):
        """ Constructor method. Loads all puzzle pieces with provided prefix.

        Args:
            directory (str): directory storing puzzle pieces
            prefix (str): prefix of puzzle piece filenames (created using create_jigsaw_pieces.py)
        """

        images = []
        for (_, _, filenames) in os.walk(directory):
            filenames = sorted(filenames) # 0.png should be start
            for filename in filenames:
                if filename.endswith('.png'):
                    images.append(cv2.imread(directory + "/" + filename, cv2.IMREAD_UNCHANGED))
        return cls(images, verbose, out_dir)
    
    def reset(self, full=False):
        self.edge_maps = [{}, {}]
        self.merge_history = []
        self.whiteboard = None
        del self.blueprint  # blueprint for image reconstruction
        del self.images

        if full:
            accuracy = self.piece_sz
        else:
            accuracy = self.piece_sz // 2
        Config.samples = np.linspace(Config.overlap, Config.overlap + self.piece_sz, accuracy, dtype=np.int32)[2:-2]
        Config.samples_reversed = self.img_size - 1 - Config.samples

    def assemble(self):
        """
        Assemble puzzle pieces back into the original image using Prim's Minimum Spanning Tree algorithm with
        Linked Hashmap implementation of the Priority Queue.

        Algorithm:
            1. Initialize the 3D similarity matrix between each puzzle piece.
            2. Initialize a ConstructionBlueprint object to keep track of the merged pieces.
            3. Initialize a LinkedHashmapPriorityQueue object to store the puzzle pieces that will be merged next.
            4. Add the first puzzle piece to the priority queue.
            5. Loop through the priority queue until it is empty:
                a. Dequeue the puzzle piece with the highest score from the priority queue.
                b. Add the puzzle piece to the ConstructionBlueprint object.
                c. Find the best fit puzzle pieces for all possible adjacent positions.
                d. Add the best fit puzzle pieces to the priority queue.

        Returns:
            None
        """


        def _find_best_sim_matrix_index4(neighbours):
            """
            Optimized function to process data from a DataFrame based on tuples of (direction, id, side),
            with handling for null values. Finds the smallest value and the difference with the second smallest.

            Args:
                neighbours: List of tuples (direction, id, side).

            Returns:
                tuple: Difference between the smallest and second smallest, and the index of the smallest.
            """

            data_list = []
            for direction, neighbour, edge in neighbours:
                index = (neighbour.id, edge)

                if index in self.sim_matrix.index:
                    data = self.sim_matrix.loc[index]
                elif index in self.sim_matrix.columns:
                    data = self.sim_matrix[index]
                else:
                    continue

                valid_data = data[[not self.images[i].valid for i in data.index.get_level_values(0)]]

                updated_indices = [(idx[0], (direction - idx[1]) % 4) for idx in valid_data.index]
                data_list.append(pd.Series(valid_data.values, index=updated_indices))

            if not data_list:
                raise KeyError
            # Combine, filter out nulls, sum
            summed_data = pd.concat(data_list, axis=1).dropna().sum(axis=1)

            # Handle cases based on data length
            if len(summed_data) == 1:
                return *summed_data.index[0], summed_data.iloc[0]
            else:
                # Calculate and return the score and index of the smallest value
                bigest_2 = summed_data.nlargest(2)
                score = bigest_2.iloc[0] - bigest_2.iloc[1]
                index = bigest_2.idxmax()
                return *index, score
            


        def _find_best_sim_matrix_index(neighbours):
            """
            Optimized function to process data from a DataFrame based on tuples of (direction, id, side),
            with handling for null values. Finds the smallest value and the difference with the second smallest.

            Args:
                neighbours: List of tuples (direction, id, side).

            Returns:
                tuple: Difference between the smallest and second smallest, and the index of the smallest.
            """
            valid_data = None
            for direction, neighbour, edge in neighbours:
                index = neighbour.id * 4 + edge

                if index in self.sim_matrix.index:
                    data = self.sim_matrix.loc[index]
                elif index in self.sim_matrix.columns:
                    data = self.sim_matrix[index]
                else:
                    continue
                filtered_data = data[[not self.images[id].valid for id in data.index//4]]
                filtered_data.index = modulus_addition(filtered_data.index, direction)
                    
                if valid_data is None:
                    valid_data = filtered_data
                else:
                    aligned_series1, aligned_series2 = valid_data.align(filtered_data, join='inner')
                    valid_data = aligned_series1 + aligned_series2

            # Handle cases based on data length
            if len(valid_data) == 1:
                idx = valid_data.index[0]
                return idx//4, idx%4, valid_data.iloc[0], len(neighbours)
            else:
                # Calculate and return the score and index of the smallest value
                bigest_2 = valid_data.nlargest(2)
                score = bigest_2.iloc[0] - bigest_2.iloc[1]
                index = bigest_2.idxmax()
                return index//4, index%4, score, len(neighbours)

        def modulus_addition(index, rotation):
            return (index & ~3) | ((rotation - index) & 3)



        def _gather_viable_options(neighbours):
            """
            Gather all viable puzzle piece options based on neighbours.

            Args:
                neighbours (list): List of neighbour details.

            Returns:
                set: A set of viable puzzle piece options.
            """
            options = set()
            for dir, neighbour, edge in neighbours:
                type_ = neighbour.types[edge] ^ 1
                matching_edge = neighbour.calc_matching_edge(edge)

                viables = self.edge_maps[type_][matching_edge]

                # Options gets the intersect of the viables lists, whilst considering rotation
                if options:
                    options = {(id, rot) for id, rot in options 
                                if id in viables and (dir - rot) % 4 in viables[id]}
                else:
                    options = {(id, (dir - side) % 4) for id, sides in viables.items() 
                                if not self.images[id].is_valid() for side in sides}

                if neighbour.id == 113:
                    print("dir", dir)
                    print("edge", edge)
                    print(viables)
                    print(neighbour.types[edge])
            return options


        def _evaluate_best_option(options, neighbours):
            """
            Evaluate and return the best option based on the lowest score.

            Args:
                options (set): Viable puzzle piece options.
                neighbours (list): List of neighbour details.

            Returns:
                tuple: Best puzzle piece option with its score.
            """
            max_score_2 = 0
            best_candidate = [None, None, 0]

            for piece_id, rotation in options:
                score = 0
                for direction, neighbour, edge in neighbours:
                    score += helpers.edge_similarity(
                        neighbour.get_color_edge(edge),
                        self.images[piece_id].get_color_edge((direction - rotation) % 4)
                    )

                if score > best_candidate[2]:
                    max_score_2 = best_candidate[2]
                    best_candidate = [piece_id, rotation, score]
                elif score > max_score_2:
                    max_score_2 = score

            best_candidate[2] = best_candidate[2] - max_score_2
            return *best_candidate, len(neighbours)


        def _best_fit_piece_at(x, y):
            """
            Find the puzzle piece that can be most naturally stitched at the given position.

            Args:
                y: int, the row position.
                x: int, the column position.

            Returns:
                The PuzzlePiece object that is the best fit.
            """
            neighbours = self.blueprint.get_active_edges(x, y)
            if not self.check_edge_path:
                return _find_best_sim_matrix_index(neighbours)

            viable_options = _gather_viable_options(neighbours)

            if len(viable_options) == 1:
                return *viable_options.pop(), 1, len(neighbours)
            elif len(viable_options) == 0:
                raise KeyError("No piece connection found.")
            else:
                return _evaluate_best_option(viable_options, neighbours)


        def _dequeue_and_merge():
            """
            Dequeue the puzzle piece with the highest score from the priority queue
            and merge it into the ConstructionBlueprint object.
            Then remove all duplicate puzzle pieces from the priority queue.

            Returns:
                The PuzzlePiece object that was dequeued and merged.
            """
            nonlocal p_queue
            piece, duplicates = p_queue.dequeue_and_remove_duplicate_ids()
            piece_id, rotation, position = piece
            self.blueprint.activate_position(self.images[piece_id], rotation, position)
            if self.verbose:
                self.save_progress(self.images[piece_id])
            self.pieces_placed += 1
            print("image merged: ", piece_id, "\t",
                  self.pieces_placed, "/", self.img_cnt, flush=True)
            self.merge_history.append(piece)
            return piece, duplicates


        def _enqueue_all_frontiers(empty_positions):
            """
            For all next possible puzzle piece placement positions,
            find the best fit piece at each position and append them puzzle pieces to the priority queue.

            Args:
                frontier_pieces_list:
                    List of PuzzlePiece objects representing the positions of the puzzle pieces on the
                    frontier of the ConstructionBlueprint.

            Returns:
                None
            """
            nonlocal p_queue
            for pos in empty_positions:
                if self.blueprint.is_inactive(*pos): # check if needed ever...?
                    p_queue.enqueue(_best_fit_piece_at(*pos), pos)

        self._compute_edge_hashmap()


        if self.most_popular_edge * 3 <= self.edge_cnt - self.side_edge_cnt:
            self.check_edge_path = True
            self.sim_matrix = None
        else:
            self._compute_similarity_matrix()
            self.check_edge_path = False
            del self.edge_maps # I dont think I use this again



        # initialization.

        s_time = time.time()
        # replace with self.height_width tuple which is which
        self.blueprint = PuzzleBlock(self.height_width)
        p_queue = LinkedHashmapPriorityQueue(self.img_cnt)

        # add the first puzzle piece to the priority queue
        p_queue.enqueue((0, 0, 1, 0), (0, 0))

        # MST assembly algorithm loop
        while not p_queue.is_empty():
            # do not consider a position that's already activated by the blueprint
            if not self.blueprint.validate_position(*p_queue.peek()):
                p_queue.dequeue()
                continue
            # dequeue puzzle piece from the priority queue, and merge it towards the final image form.
            piece, duplicates = _dequeue_and_merge()
            # add the best fit puzzle pieces at all frontier positions to the priority queue
            _enqueue_all_frontiers(self.blueprint.get_inactive_neighbors(*piece[2])
                                   + [d[2] for d in duplicates])

        print("MST assembly algorithm:", time.time() - s_time, "seconds")


    def save_assembled_image2(self, filepath):
        """
        save the reconstructed image to a file

        Args:
            filepath (str): The path and filename to save the image to.
        """
        overlap = Config.overlap
        
        im_size = self.raw_imgs[0].shape[0]
        net_size = im_size - overlap
        total_width = self.blueprint.w * net_size + overlap
        total_height = self.blueprint.h * net_size + overlap
        whiteboard = np.zeros((total_width, total_height, 4), dtype=np.uint8)
        
        for img in self.images:
            if img.is_valid():
                x = img.x * net_size
                y = img.y * net_size
                rotated_img = np.rot90(img.data, -img.rotation)
                # Efficient pasting using NumPy slicing and masking
                alpha_mask = rotated_img[:, :, 3] != 0
                target_slice = whiteboard[y:y+im_size, x:x+im_size]
                np.copyto(target_slice, rotated_img, where=alpha_mask[..., None])
        
        cv2.imwrite(filepath + ".png", whiteboard[overlap:-Config.overlap,
                                                  overlap:-overlap, :])

    def save_assembled_image(self, filepath):
        """
        save the reconstructed image to a file

        Args:
            filepath (str): The path and filename to save the image to.
        """
        overlap = Config.overlap
        
        total_width = self.blueprint.w * self.piece_sz + 2 * overlap
        total_height = self.blueprint.h * self.piece_sz + 2 * overlap
        whiteboard = np.zeros((total_height, total_width, 4), dtype=np.uint8)
        
        for img in self.images:
            if img.is_valid():
                x = img.x * self.piece_sz
                y = img.y * self.piece_sz
                rotated_img = np.rot90(img.data, -img.rotation)
                # Efficient pasting using NumPy slicing and masking
                alpha_mask = rotated_img[:, :, 3] != 0
                target_slice = whiteboard[y:y+self.img_size, x:x+self.img_size]
                np.copyto(target_slice, rotated_img, where=alpha_mask[..., None])
        
        cv2.imwrite(filepath + ".png", whiteboard[overlap:-Config.overlap,
                                                  overlap:-overlap, :])
        

    def save_progress(self, img):
        """
        save the reconstructed image to a file

        Args:
            filepath (str): The path and filename to save the image to.
        """
        
        overlap = Config.overlap
        
        if self.whiteboard is None:
            total_width = self.blueprint.w * self.piece_sz + 2 * overlap
            total_height = self.blueprint.h * self.piece_sz + 2 * overlap
            self.whiteboard = np.zeros((total_height, total_width, 4), dtype=np.uint8)
        
        x = img.x * self.piece_sz
        y = img.y * self.piece_sz
        rotated_img = np.rot90(img.data, -img.rotation)
        # Efficient pasting using NumPy slicing and masking
        alpha_mask = rotated_img[:, :, 3] != 0
        target_slice = self.whiteboard[y:y+self.img_size, x:x+self.img_size]
        np.copyto(target_slice, rotated_img, where=alpha_mask[..., None])
        cv2.imwrite(f"{self.out_dir}{self.pieces_placed}.png", 
                    self.whiteboard[overlap:-Config.overlap,
                    overlap:-overlap, :])
        



    def start_assembly_animation(self, show_spanning_tree, interval_millis=200):
        """
        Show an animation of the assembly process after it is complete.

        Args:
            show_spanning_tree (bool): if True, the animation will display the MST used during assembly.
            interval_millis (int): The interval (in milliseconds) between animation frames.
        """
        vis.start_assembly_animation(self.blueprint, self.merge_history, self.raw_imgs_unaligned,
                                     self.raw_imgs, show_spanning_tree, interval_millis)
        

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        private methods
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    def _compute_edge_hashmap(self):
        self.images = [Image(id, img) for id, img in enumerate(self.raw_imgs)]
        self.holes = [(id, side) for id, image in enumerate(self.images) for side in range(4) if image.types[side] == 0]
        self.knobs = [(id, side) for id, image in enumerate(self.images) for side in range(4) if image.types[side] == 1]
        self.side_edge_cnt = self.edge_cnt - len(self.holes) - len(self.knobs)

        for i, edges in enumerate([self.holes, self.knobs]):
            for id, side in edges:
                self.edge_maps[i].setdefault(tuple(self.images[id].edges[side]), {}).setdefault(id, []).append(side)

        self.most_popular_edge = max(
            len(edges)
            for edge_dict in self.edge_maps
            for edges in edge_dict.values()
        )
        # array with all edges value in, so that you can map 
        # calculate non straight edges
        self.height_width = helpers.get_width_and_height(self.img_cnt, self.side_edge_cnt)

        return # for now


    def convert_to_color(self, image, edge, side):
        if side == TOP:
            return image[edge, self.samples, :]
        elif side == RIGHT:
            return image[self.samples, -edge-1, :]
        elif side == LEFT:
            return image[-edge-1, self.samples, :]
        else: # BOTTOM
            return image[self.samples, edge, :]


    def _compute_color_edges(self):
        for id in range(self.img_cnt):
            self.images[id].calc_all_color_edges()


    def _compute_similarity_matrix4(self):
        """
            Computes the similarity matrix for all image pairs by considering all
            possible combinations of stitching directions and orientations.
        """
        s_time = time.time()
        # initialize
        type_count = (self.edge_cnt - self.side_edge_cnt) // 2
        holes = [4*id+side for id, side in self.holes]
        knobs = [4*id+side for id, side in self.knobs]

        if not (len(knobs) == len(holes) == type_count):
            raise KeyError

        self.sim_matrix = pd.DataFrame(np.zeros((type_count, type_count)),
                                        index=pd.MultiIndex.from_tuples(knobs),
                                        columns=pd.MultiIndex.from_tuples(holes))

        cols = np.array([self.images[col[0]].get_color_edge(col[1]) for col in holes])
        rows = np.array([self.images[row[0]].get_color_edge(row[1]) for row in knobs])

        squared_diffs = (rows[:, np.newaxis, :] - cols[np.newaxis, :, :]) ** 2
        self.sim_matrix[:] = np.mean(squared_diffs, axis=(2, 3))
            
        print("similarity matrix construction:", time.time() - s_time, "seconds")


    def _compute_similarity_matrix(self):
        """
            Computes the similarity matrix for all image pairs by considering all
            possible combinations of stitching directions and orientations.
        """
        s_time = time.time()
        # initialize
        type_count = (self.edge_cnt - self.side_edge_cnt) // 2
        holes = [k*4+i for k, image in enumerate(self.images) for i in range(4) if image.types[i] == 0]
        knobs = [k*4+i for k, image in enumerate(self.images) for i in range(4) if image.types[i] == 1]

        self.sim_matrix = pd.DataFrame(np.zeros((type_count, type_count)),
                                        index=knobs,
                                        columns=holes)

        cols = np.array([self.images[col//4].get_color_edge(col%4) for col in holes])
        rows = np.array([self.images[row//4].get_color_edge(row%4) for row in knobs])

        squared_diffs = (rows[:, np.newaxis, :] - cols[np.newaxis, :, :]) ** 2
        self.sim_matrix[:] = np.mean(squared_diffs, axis=(2, 3))
            
        print("similarity matrix construction:", time.time() - s_time, "seconds")

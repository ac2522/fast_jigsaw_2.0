"""assembler_data_structures.py
data structures for assembler.py

CellData: representation of image cell including id, orientation and x,y position
CellBlock: blueprint for image assembly.
LHashmapPriorityQueue: linked-hashmap implementation of MST priority queue.
"""

import numpy as np

# direction ENUM (id, y delta, x delta)
DIR_ENUM = {
    'u': (0, -1, 0),
    'r': (1, 0, 1),
    'd': (2, 1, 0),
    'l': (3, 0, -1),
}

TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3
HOLE, KNOB, FLAT = 0, 1, 2

import numpy as np


class Config:
    # Configuration parameters for edge processing
    overlap = None
    half = None
    samples = None
    samples_reversed = None
    IDX_S = None # 20% of Samples
    IDX_K = None # Knob location

    def get_samples(reversed=False):
        if reversed:
            return Config.samples_reversed
        else:
            return Config.samples



class Image:
    c = Config

    def __init__(self, img_id, img, rotation=0):
        """
        Initializes an Image object.

        Args:
            img_id (int): Unique identifier for the image.
            img (ndarray):  Numpy array representing the image data.
            rotation (int, optional): Initial rotation of the image (in multiples of 90 degrees). 
                Defaults to 0.
        """
        self.id = img_id
        self.rotation = rotation
        self.valid = False # Flag indicating if piece has been placed in the puzzle
        self.edges = {} # Stores processed edges (path) of each side
        self.types = {}     # Stores type of each edge (HOLE, KNOB, FLAT)
        self.color_edges = {}  # Stores color information for each edge
        self.y, self.x = -1, -1  # Position of the image within the puzzle

        # Processing and storing edges
        o = Image.c.overlap
        h = Image.c.half
        self.types[0], self.edges[0] = self.process_edge(img[:, :, 3], img[[o+1, o-2], h, 3], 0)
        self.types[1], self.edges[1] = self.process_edge(img[:, ::-1, 3], img[h, [-o-2, -o+1], 3], 1)
        self.types[2], self.edges[2] = self.process_edge(img[::-1, :, 3], img[[-o-2, -o+1], h, 3], 2)
        self.types[3], self.edges[3] = self.process_edge(img[:, :, 3], img[h, [o+1, o-2], 3], 3)


        self.color_edges = {0: None, 1: None, 2: None, 3: None}
        self.data = img

    def process_edge(self, image, edge_type, side):
        """
        Processes a puzzle piece edge and normalizes its representation.

        Args:
            image (ndarray): Image data (transparency layer).
            edge_type (ndarray): Type of edge along a specific slice.
            side (int): The side of the image being processed (0: TOP, 1: RIGHT, 2: BOTTOM, 3: LEFT).

        Returns:
            tuple: (edge_type, normalized_path)
                edge_type: Type of edge (HOLE, KNOB, FLAT)
                normalized_path: Array representing the edge path 
        """
    
        def find_first_non_zero(alpha, axis, is_knob=False):
            """
            Finds the first non-zero index along a specified axis, considering knobs/holes for overlap correction.

            Args:
                alpha (ndarray): Input array.
                axis (int): Axis to search along.
                is_knob (bool, optional): Flag indicating if the edge is a knob. Defaults to False.

            Returns:
                ndarray: Array of indices of the first non-zero element along the given axis.
            """
            if is_knob:
                if axis == 1:
                    return np.concatenate([np.argmax(alpha[:Image.c.IDX_S, :], axis=1),
                                        alpha.shape[1] - np.argmin(alpha[Image.c.IDX_S:-Image.c.IDX_S, ::-1], axis=1),
                                        np.argmax(alpha[-Image.c.IDX_S:, :], axis=1)])
                else:
                    return np.concatenate([np.argmax(alpha[:, :Image.c.IDX_S], axis=0),
                                        alpha.shape[0] - np.argmin(alpha[::-1, Image.c.IDX_S:-Image.c.IDX_S], axis=0),
                                        np.argmax(alpha[:, -Image.c.IDX_S:], axis=0)])
            return np.argmax(alpha, axis=axis)
        
        if edge_type[0] == 0:
            type_, idx = HOLE, Image.c.half
        elif edge_type[1] != 0:
            type_, idx = KNOB, Image.c.IDX_K
        else:
            return (FLAT, None)
        
        samples = Image.c.get_samples((side in [BOTTOM, LEFT]) ^ type_)

        if side in [TOP, BOTTOM]:
            return type_, find_first_non_zero(image[:idx, samples], 0, type_==KNOB)
        else: # RIGHT, LEFT
            return type_, find_first_non_zero(image[samples, :idx], 1, type_==KNOB)


    def get_edge(self, side):
        """
        Retrieves the pre-processed edge path for a specified side.

        Args:
            side (int): The side of the image (0: TOP, 1: RIGHT, 2: BOTTOM, 3: LEFT).

        Returns:
            ndarray: The normalized edge path, or None if the edge doesn't exist.
        """
        return self.edges[side]

    def calc_color_edge(self, side):
        """
        Extracts color information along a specified edge of the image.

        Args:
            side (int): The side of the image (0: TOP, 1: RIGHT, 2: BOTTOM, 3: LEFT).
            img (ndarray): The full image data.
        """
        samples = Image.c.get_samples((side in [BOTTOM, LEFT]) ^ self.types[side])

        if side == TOP:
            self.color_edges[side] = self.data[self.edges[TOP], samples, :3]/ 25
        elif side == RIGHT:
            self.color_edges[side] = self.data[samples, -self.edges[RIGHT] - 1, :3]/ 25
        elif side == BOTTOM:
            self.color_edges[side] =  self.data[-self.edges[BOTTOM] - 1, samples, :3]/ 25
        else:  # LEFT
            self.color_edges[side] = self.data[samples, self.edges[LEFT], :3] / 25

    def calc_all_color_edges(self):
        """
        Calculates and stores color information for all edges of the image.

        Args:
            img (ndarray): The full image data.
        """
        
        for side in range(4):
            if self.edges[side] is not None:
                self.calc_color_edge(side)

    def is_valid(self):
        """
        Checks if the image has been assigned a valid position in the puzzle.

        Returns:
            bool: True if the image has a valid position, False otherwise.
        """
        return self.valid

    def get_color_edge(self, side):
        """
        Retrieves the pre-calculated color information for a specified edge.

        Args:
            side (int): The side of the image (0: TOP, 1: RIGHT, 2: BOTTOM, 3: LEFT).

        Returns:
            ndarray: The color information for the edge, or None if not calculated.
        """
        if self.color_edges[side] is None:
            self.calc_color_edge(side)

        return self.color_edges[side]

    def calc_matching_edge(self, side):
        """
        Calculates the complementary edge (the 'matching' edge on the neighboring piece) 
        for a specified side.

        Args:
            side (int): The side of the image (0: TOP, 1: RIGHT, 2: BOTTOM, 3: LEFT).

        Returns:
            ndarray: The normalized path of the complementary edge.
        """
        return tuple(Image.c.overlap * 2 - self.edges[side])
    
    def needs_reversing(self, side):
        return (side in [2, 3]) ^ (self.types[side] == 0)

    def is_edge(self):
        """
        Determines if the image is an edge piece of the puzzle.

        Returns:
             bool: True if the image has both a bottom and right edge, False otherwise.
        """
        return (self.types[(RIGHT - self.rotation) % 4] == 2 or  # Check for RIGHT edge
                self.types[(BOTTOM - self.rotation) % 4] == 2)   # Check for BOTTOM edge
    


    def __str__(self):
        if self.is_valid():
            return f"{self.id}: ({self.x},{self.y})"
        else:
            return str(self.id)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Image):
            return NotImplemented
        return self.id == other.id and self.rotation == other.rotation and \
            self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.score < other.score


    def tostring(self):
        """
        Returns a string representation of the puzzle piece
        """
        return ("{id: " + str(self.id) + ", rotation: " + str(self.rotation) +
                ", pos: (" + str(self.x) + ", " + str(self.y) + ")}")
    

    def set(self, valid=False, rotation=-1, x=-1, y=-1):
        """
        Sets the attributes of the puzzle piece
        """
        self.valid=valid

        if rotation != -1:
            self.rotation = rotation
        if y != -1 and x != -1:
            self.y, self.x = y, x
        return self
    
    def get_facing_edge(self, side):
        """
        Args: side (int 0-3) is the neigbours side index 
        returns image id, and the edge index that is neighbouring a certain direction
        """
        # what type of side is this: (side - self.rotation) % 4
        # what type of side is this attatched to: (2 + side - self.rotation) % 4
        return (2 + side - self.rotation) % 4

    def pos(self):
        """
        position of the cell within PuzzlrlBlock
        """
        return self.x, self.y


class PuzzleBlock:
    """
    Blueprint for assembling a jigsaw puzzle image.
    Holds 2d array of PuzzlePiece objects, each initialized with a blank image piece.

    - Positions in the array are considered "inactive"
      until they are activated by pasting a valid PuzzlePiece object via the activate_position() method.
    - A position can only be activated if it is adjacent to another activated position,
      and if the resulting assembly of activated positions does not exceed
      the maximum allowed size specified by max_h and max_w.

    Attributes:
        h (int): actual height (in pieces high)
        w (int): actual width (in pieces wide)
        max_hw (int): maximum allowed number of images in a row/column, either actual height or width
        min_hw (int): minimum allowed number of images in a row/column, either actual width or height
        data (2d list of PuzzlePiece objects): the 2D array holding the PuzzlePiece objects
        bottom, top, left, right (int): the current boundaries of the activated positions in the array

    Methods:
        get_active_neighbors(y: int, x: int) -> list of PuzzlePiece: returns all neighboring active PuzzlePiece objects
        get_inactive_neighbors(y: int, x: int) -> list of PuzzlePiece: returns all neighboring inactive PuzzlePiece objects
        validate_position(y: int, x: int) -> bool: checks if position y, x can be activated
        activate_position(piece: PuzzlePiece): activates position y, x by pasting PuzzlePiece object at y, x
        block_size() -> height, width: returns the current blueprint size of activated positions
    """

    def __init__(self, height_width, verbosity=0, overlap=0):

        self.found_height = (height_width[0] == height_width[1])
        # width = height
        if self.found_height:
            self.h = height_width[0]
            self.w = height_width[0]
        else: 
            self.h = max(height_width)
            self.w = max(height_width)
            self.max_hw = max(height_width)
            self.min_hw = min(height_width)

        # huh???
        self.verbosity = verbosity
        self.overlap = overlap
        self.empty = True
        # initializw 2D array for Image objects
        self.data = [[None for _ in range(self.w)] for _ in range(self.h)]


        if self.verbosity == 2:
            global Image
            from PIL import Image

    def set(self, x, y, img):
        self.data[y][x] = img

    def get(self, x, y):
        return self.data[y][x]
    
    def is_active(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h and self.data[y][x] is not None
    
    def is_inactive(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h and self.data[y][x] is None

    def get_active_neighbors(self, x, y):
        """
        Returns a list of neighboring active PuzzlePiece objects.

        Args:
            y (int): the y-coordinate of the position
            x (int): the x-coordinate of the position

        Returns:
            List[PuzzlePiece]: a list of neighboring active PuzzlePiece objects
        """

        adj = []
        for dirc in DIR_ENUM.values():
            if self.is_active(x + dirc[2], y + dirc[1]):
                adj.append(self.data[y + dirc[1]][x + dirc[2]]) # , dirc[0]
        return adj

    def get_active_edges(self, x, y):
        """
        Returns a list of all neighboring image edge.

        Args:
            y (int): the y-coordinate of the position
            x (int): the x-coordinate of the position

        Returns:
            List[direction, (Image.id, edge index) ]: a list of neighboring Image edges
        """

        adj = []
        for dirc in DIR_ENUM.values():
            x_ = x + dirc[2]
            y_ = y + dirc[1]
            if self.is_active(x_, y_):
                #adj.append((dirc[0], self.data[y + dirc[1]][x + dirc[2]].get_info(dirc[0])))
                adj.append((dirc[0], self.data[y_][x_], self.data[y_][x_].get_facing_edge(dirc[0])))
        
        return adj

    def get_inactive_neighbors(self, x, y):
        """
        Returns a list of neighboring inactive Image objects.
        """
        adj = []
        for dirc in DIR_ENUM.values():
            if self.is_inactive(x + dirc[2], y + dirc[1]):
                adj.append((x + dirc[2], y + dirc[1]))
        return adj


    def validate_position(self, x, y, neighbors):
        """
        check if the given position y, x can be activated (i.e., a puzzle piece can be placed there).
        A position is valid if it is not already occupied and has at least one adjacent activated position.
        """
        return (self.is_inactive(x, y) and
                neighbors == len(self.get_active_neighbors(x, y))) or self.empty


    def activate_position(self, image, rotation, position):
        """
        Activate the given puzzle piece by placing it in the corresponding position in the block.

        @Parameters
        image (Image)
        """
        image.set(True, rotation, *position)
        print(f"ACTIVATED: {image.id} at ({position}) - rot: {rotation}")
        
        self.data[image.y][image.x] = image

        if not self.found_height and image.is_edge():
            self.found_height = True
            if image.y > min(self.max_hw, self.min_hw):
                self.h, self.w = self.max_hw, self.min_hw
            else:
                self.h, self.w = self.min_hw, self.max_hw
        if self.empty:
            self.empty = False


    def print_state(self):
        """
        Print the current state of the puzzle based on the verbosity level set. 
        There are three verbosity levels: 0 (no output), 1 (print locations), and 2 (print images).
        """

        if self.verbosity == 1:
            self._print_puzzle_as_grid()
        elif self.verbosity == 2:
            self._print_puzzle_as_image()
        

    def _print_puzzle_as_grid(self):
        """
        Outputs the ID and orientation for each valid piece, or 'Empty' for missing pieces.
        """

        for row in self.data:
            for piece in row:
                if piece.is_valid():
                    print(f"ID: {piece.img_id}, Rot: {piece.orientation}", end=" ")
                else:
                    print("Empty", end=" ")
            print()

    
    def _print_puzzle_as_image(self):
        """
        Create and output an image representing the current state of the puzzle. 
        This method calculates the total size of the puzzle and pastes each piece at its correct position.
        """

        # Determine the size of each puzzle piece (assuming all pieces are of the same size)
        piece_height, piece_width = self.data[0][0].image.size

        # Calculate the dimensions of the output image
        height = (self.top - self.bottom + 1) * piece_height - self.data[0][0].overlap
        width = (self.right - self.left + 1) * piece_width - self.data[0][0].overlap

        # Create an empty image with transparency
        output_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        pieces_count = 0
        # Iterate over the grid and paste each piece onto the output image
        for y in range(self.bottom, self.top + 1):
            for x in range(self.left, self.right + 1):
                piece = self.get(x, y)
                if piece.is_valid():
                    # Calculate the position to paste this piece
                    pos_x = (x - self.left) * (piece_width - piece.overlap)
                    pos_y = (y - self.bottom) * (piece_height - piece.overlap)

                    im = piece.image
                    #im = rotate(piece.image, piece.orientation)

                    # Check the image mode and paste accordingly
                    if im.mode == 'RGBA':
                        mask = im.split()[3]  # Use the alpha channel as a mask
                        output_img.paste(im, (pos_x, pos_y), mask)
                    else:
                        output_img.paste(im, (pos_x, pos_y))

                    pieces_count += 1

        #output_img.save(f"step_{pieces_count}.png")
        output_img.show()

class LinkedHashmapPriorityQueue:
    """
    Linked Hashmap implementation of Priority Queue
    {key(int): val(Comparable object)}

                        max-heap vs linked-hashmap
    enqueue()           O(logN)     O(N)
    dequeue()           O(logN)     O(1)
    search_by_key()     O(N)        O(1)

    Attributes:
        hashmap (dict)
        ll_head (Node)

    Methods:
        is_empty() -> bool
        peek() -> comparable object
        enqueue(key, val) -> void
        dequeue() -> comparable object
        dequeue_and_remove_duplicate_ids() -> comparable object, list of comparable objects

    """

    class Node:
        """
        doubly linked list node with key-val pair

        Attributes:
            key (int)
            rot (int)
            next (Node)
            prev (Node)
        """

        def __init__(self, best_choice, position):
            self.id, self.rot, self.score, self.neighbours = best_choice
            self.pos = position
            
            self.next = None
            self.prev = None

        def get_data(self):
            return self.id, self.rot, self.pos
        
        def check_viable(self):
            return *self.pos, self.neighbours

        def __str__(self):
            return str(self.get_data())

        def __repr__(self):
            return self.__str__()

    def __init__(self, max_size):
        self.hashmap = {i: [] for i in range(max_size)}
        self.ll_head = None  # linked list head

    def is_empty(self):
        """
        Returns:
            empty (bool)
        """
        return self.ll_head is None

    def peek(self):
        """
        Returns:
            val (Object)
        """
        return self.ll_head.check_viable()

    def enqueue(self, best_choice, position):
        """
        Args:
            img_id (int):
            rotation (int) 0-3:
            score (float):
            position (tuple) (x,y):
        """
        node = self.Node(best_choice, position)
        self.hashmap[node.id].append(node)

        if self.ll_head is None:
            self.ll_head = node
            return
        if node.score >= self.ll_head.score:
            node.next = self.ll_head
            self.ll_head = node
            node.next.prev = node
            return

        ptr = self.ll_head
        while ptr.next is not None:
            if node.score >= ptr.next.score: # bigger is better
                node.next = ptr.next
                node.prev = ptr
                ptr.next = node
                node.next.prev = node
                return
            ptr = ptr.next
        ptr.next = node
        node.prev = ptr

    def dequeue(self):
        """
        Returns:
            node_val (object)
        """
        if not self.ll_head:
            return None
        return_node = self.ll_head

        self.ll_head = self.ll_head.next
        if return_node.next:
            return_node.next.prev = None
        self.hashmap[return_node.id].remove(return_node)
        return return_node.get_data()

    def dequeue_and_remove_duplicate_ids(self):
        """
            dequeue top priority node & remove all nodes with the same key from the l_hashmap.
            runtime = O(N)
        """
        if not self.ll_head:
            return None
        return_node = self.ll_head

        duplicates_list = []
        for duplicate in self.hashmap[return_node.id]:
            duplicates_list.append(duplicate.get_data())
            if duplicate == self.ll_head:
                self.ll_head = self.ll_head.next
                if duplicate.next:
                    duplicate.next.prev = None
            else:
                duplicate.prev.next = duplicate.next
                if duplicate.next:
                    duplicate.next.prev = duplicate.prev
        self.hashmap[return_node.id] = []
        return return_node.get_data(), duplicates_list

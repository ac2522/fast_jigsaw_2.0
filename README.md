# fast_jigsaw_2.0

Started out as a fork of <br>https://github.com/hj2choi/fast_jigsaw_puzzle_solver</br> 

# Fast Jigsaw Puzzle Solver with unknown orientation
- Converts images into <b>N</b> (row x col) pieces, with knobs and holes, mixes them up.</br>
- Solves the puzzle, returning it to original image in <b>O(N<sup>2</sup>P)</b> runtime. Where P is pixel size of width or height.</br>
![demo_anim](https://github.com/ac2522/ac2522_images/blob/main/puzzle_solving.gif)</br>
<i>Disclaimer: orientation of the final image is random. Successful reconstruction is not always guaranteed.</i>

### Features
  - Euclidean distance metric for image boundary matching
  - Uses relative comparison algorithm,
  - Prim's <b>Minimum Spanning Tree</b> algorithm with Linked-Hashmap implementation of the Priority Queue<br>
  - Contour path calcualtion and a contour hashmap
  - Can solve solid color puzzles.
  - Combines 2 algorithms: to match contour and color similarity. 

### Dependencies
python 3.7+  
numpy 1.16+  
opencv-python 4.1.2+  

## Execution guide
### Quick demo with animation
```bash
pip install -r requirements.txt
bash demo.sh
```  

#### create_jigsaw_pieces.py: read and slice image into equal-sized jigsaw pieces and apply random set of transformations.
```bash
create_jigsaw_pieces.py [OPTION] ${image_filename} ${piece_size}
```
-v: *increase verbosity*</br>
--method: *for determining knob edge contour paths: uniform, classic, random*</br>
--size_mismatch: *If image size does not divide exactly by pieces: crop, stretch, black, white*</br>
<img src="https://github.com/ac2522/ac2522_images/blob/main/jigsaw_pieces.png" width="560" title="jigsaw pieces">
</br>

#### solve_puzzle.py: reconstruct puzzle pieces back to original image
```bash
solve_puzzle.py [OPTION] ${keystring}
```
-v: *increase verbosity*<br/>
-a: *show animation* - outputs intermediate steps, I will modify this so it outputs as gif<br/>
<img src="https://github.com/ac2522/ac2522_images/blob/main/intermediate_steps.png" width="560" title="intermediate results">



## image reconstruction algorithm
```
I[i,t]: all unplaced puzzle pieces (image, transformation)
S[i,j,t]: all-pairs puzzle piece similarity-matrix
G[y,x]: puzzle block
Q: priority queue
R: rotations: values 0:3
E[i, t] edge hashmap,  [{(edge contour): {image id: [side]}}, {(edge contour): {image id: [side]}}]
```
<img src="https://github.com/ac2522/ac2522_images/blob/main/full_algorithm.png" width="1000">
<sup>7</sup> If len(P)==1: it sets w.im=P[0]. Given the rarity of duplicates, this should most often be the case. (2N-1)/pixels<sup>log(pixels)</sup>

## Time complexity analysis
<b>N</b>: number of images (puzzle pieces)</br>
<b>C</b>: total cache miss (number of duplicate puzzle pieces to be removed from queue)</br>
in all cases, <b>C = O(N)</b></br>
<b>P</b>: length of linear edge in pixels (P does not need to take into account knob or hole)
Regular knobs are generated so that every knob is the same, every hole is the same. A randomly generated knob has the likelihood of becoming a regular knob puzzle with probability:
<img src="https://github.com/ac2522/ac2522_images/blob/main/Random_duplicates.png" width="100">

| Operations \ Algorithms       | hj2choi Ver. | Square edges | Regular knobs | Random knobs |
|:------------------------------| :---: | :---: | :---: | :---: |
| <i>Load images</i>            | O(32NP<sup>2</sup>) | O(4NP<sup>2</sup>) | O(4NP<sup>2</sup>) * | O(4NP<sup>2</sup>) * |
| Calculate contours            | - | O(4N &middot; <i>log</i>(P)) | O(4NP &middot; <i>log</i>(P)) | O(4NP &middot; <i>log</i>(P)) |
| Edge hashmap                  | - | O(4N &middot; <i>log</i>(P)) | O(4N &middot; <i>log</i>(P)) | O(4N &middot; <i>log</i>(P)) |
| Similarity matrix             | O(32N<sup>2</sup>P) | O(8N<sup>2</sup> &middot; <i>log</i>(P)) | O(4N<sup>2</sup> &middot; <i>log</i>(P)) | - |
| Traverse all puzzle pieces    | O(N) | O(N) | O(N) | O(N) |
| Edge map search               | - | - | - | O(<i>log</i>(P)) ** |
| </br>Argmax ***<br>            | </br>O(32N)<br> | </br>O(8N) *<sup>4</sup><br> | </br>O(2N) *<sup>5</sup><br> | 4N &middot; <i>log</i>(P)   </br> O( 1  +  ———————) ≈ O(1)*<sup>6</sup> </br> P<sup>2<i>log</i>(P)</sub>   |
| <i>(PQueue)</i> remove by ID  | O(C) | O(C) | O(C) | <b>O(C) ≈ O(1)</b> *<sup>7</sup> |
| <i>(PQueue)</i> enqueue       | O(N) | O(N) | O(N) | ≈ O(1) |
|<br><br>Total time complexity</br></br>| O(32NP<sup>2</sup>)</br>+O(16N<sup>2</sup>)</br>+O(32(C+N<sup>2</sup>P))</br>+O(N(C+N))</br>|O(4NP<sup>2</sup>)</br>+O(2 &middot; 4N &middot; <i>log</i>(P))</br>+O(8N<sup>2</sup>)</br>+O(8(C+N<sup>2</sup>))</br>+O(N(C+N))|O(4NP<sup>2</sup>)</br>+O(4N(1+P) &middot; <i>log</i>(P))</br>+O(4N<sup>2</sup>)</br>+O(32(C+N<sup>2</sup>))</br>+O(N(C+N)|O(4NP<sup>2</sup>)</br>+O(4N(1+P) &middot; <i>log</i>(P))</br>+O(N &middot; <i>log</i>(P))</br>+O(1)</br>|
| <b>=</b>                      | O(N<sup>2</sup>P + NP<sup>2</sup>) | O(N<sup>2</sup> &middot; <i>log</i>(P) + NP<sup>2</sup>) | O(N<sup>2</sup> &middot; <i>log</i>(P) + NP<sup>2</sup>) | <b>O(NP<sup>2</sup>)</b> |
| _____________________         | | | | |
| <b>Optional:</b>              | | | | |
| <i>Print progress steps</i>   | - | <i>O(NP<sup>2</sup><sup>2</sup>)</i> | <i>O(NP<sup>2</sup><sup>2</sup>)</i> | <i>O(NP<sup>2</sup><sup>2</sup>)</i> |
| <i>Save images</i>            | <i>O(NP<sup>2</sup>)</i> | <i>O(NP<sup>2</sup>)</i> | <i>O(NP<sup>2</sup>)</i> | <i>O(NP<sup>2</sup>)</i> |


\* Worth noting that <b>O(P)</b> for regular knobs and random knobs will be greater when loading images, by <i>+ 2 &middot; max(knob_size|) </i>

**   Uses the contour path as a hash key: length log(P). It would be possible to preinitialize hashkeys length O(1), this would lead to an algorithmically insignificant number of extra collisions. So this could potentially be O(1).

***  On average for a particular piece there are roughly 2 active neighbours. Not quite sure how hj2choi's 32 was calculated here.

*<sup>4</sup> N possible pieces, 4 possible rotations, average 2 neighbours.

*<sup>5</sup>  N possible pieces, 4 possible rotations, average 2 neighbours, for every neighbour divide by 2 (combinations of knob/hole).

*<sup>6</sup>  log(p) is the number of samples and is calculated: `max(int(log2(self.piece_sz) * 4 - 10), 10)`. On the same note P isn’t actually equal to P, it’s hard to calculate the randomness of the beziers, but its closer to 0.1P. Also this assumes that there is only 1 active neighbour. So for a puzzle piece sz 100x100, you’d need approximately 3.8E31 puzzle pieces. WHich means you’d need at least 400,000,000,000 yottabytes of storage. So pretty safe rounding to O(1).

*<sup>7</sup> Duplicates are algorithmically insignificant, see *<sup>6</sup>


## Space complexity analysis
Bottlenecked by number of puzzle pieces and their size + output image size + similarity matrix: <b>4NP<sup>2</sup> + 4NP<sup>2</sup>s + 4N<sup>2</sup> &middot; <i>log</i>(P)</b>
hj2choi saves rotated/flipped copies of each puzzle pieces: <b>32NP<sup>2</sup> + 4NP<sup>2</sup></b> + 32N<sup>2</sup>P</b>
If you don't want a 'finished_image', instead you just want puzzle pieces rotated and relabled, you could load each image in turn, get contour then delete from memory (not implemented): <b>4P<sup>2</sup> + 4N<sup>2</sup> &middot; <i>log</i>(P)</b>
If no finished image' and random knobs: <b>4P<sup>2</sup> + 4N &middot; <i>log</i>(P)</b>

### references
http://chenlab.ece.cornell.edu/people/Andy/publications/Andy_files/Gallagher_cvpr2012_puzzleAssembly.pdf</br>
http://www.bmva.org/bmvc/2016/papers/paper139/paper139.pdf</br>
https://en.wikipedia.org/wiki/Prim%27s_algorithm</br>
https://en.wikipedia.org/wiki/Priority_queue</br>
https://github.com/python/cpython/blob/master/Lib/heapq.py</br>

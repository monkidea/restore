import os
import sys
import PIL
from PIL import Image, ImageStat
from array import array

# Code was originally written by Geoff Daniell
# 25/5/17 improvements to code, no change in output.

###############################################################################
# Note that pyrestore2.py is preferred over pyestore3.py since the former     #
# uses the colour quantisation algorithm built into the PIL library.  This    #
# does not lead to identical results to the gimp plug-in Restore2.py. If, for #
# any reason, it is desirable to have the same results using the stand-alone  #
#code and the gimp plug-in one can use pyestore3.py and Restore3.py.  The     #
#latter is also provided to to keep the plug-in working if in the future the  #
# colour indexed mode is removed from GIMP.                                   #
###############################################################################


#-------------------------------------------------------------------------------------}

def get_colourmap(image):
# Gets best 256 colours in image using own algorithm described in the appendix
# to restore2.pdf.  See this document for details of the extrapolation.

    colmap = my_palette(image)
# Exit with None if palette fails.
    if colmap==None: return None
# Get parameters for extrapolation.
    C_hi = [0,0,0]
    for C in [R,G,B]:
        s0 = sum(colmap[C])
        s1 = sum([i*colmap[C][i] for i in range(0,256)])
        C_hi[C] = int(0.5*(max(colmap[C]) + 3.04e-5*(3.0*s1 - 254.0*s0)))
        C_hi[C] = min([C_hi[C], 255])

    return (colmap, C_hi)

##############################################################################
# The following group of functions are used by the function my_palette which
# computes an optimum palette of 256 colours using an octree method.
##############################################################################

def Y(r,g,b):
# Computes the Y value from rgb values, used for sorting colours by brightness
    rr = expand[r]; gg = expand[g]; bb = expand[b]
    Y = rr*0.2126 + gg*0.7152 + bb*0.0722 + 1e-10
    return Y

#-------------------------------------------------------------------------------------}

def rgb2n(r, g, b):
# A colour is represented by a single integer obtained by interleaving the
# bits from the rgb values so that similar colours map onto close numbers.
# It is the  box number at level 6 computed from (truncated) rgb values.
    return ((ctab[r] << 2) | (ctab[g] << 1) | ctab[b] ) >> 6

#-------------------------------------------------------------------------------------}

def interpolate(c):
# interpolation in ctab is used to convert colour number to rgb
    i = 0; j = 128
    while j:
        if c >= ctab[i+j]: i += j
        j = j>>1
    return i

#-------------------------------------------------------------------------------------}

def n2rgb(n, level):
# Computes rgb values at centre of box from colour number. Returns tuple of 
# rgb values.

# Shift colour number to adjust for level in tree.
    nn = n << (24 - 3*level)
# Unpack in r and g parts.
    nr = (nn & 0x924924) >> 2
    ng = (nn & 0x492492) >> 1
    nb = (nn & 0x249249)
# Get lower corner value in box.  The function interpolate finds entry in ctab.
    r = interpolate(nr); g = interpolate(ng); b = interpolate(nb)
# Add half box length to get value at centre.
    mid = 0x80 >> level
    return (r | mid, g | mid, b | mid)

#-------------------------------------------------------------------------------------}

def colourfit(x, l):
# Computes measure of optimality of the fit.  Note the recursion.
    (c, n, sub_boxes) = x
    s = n/2**l
    for box in sub_boxes: s += colourfit(box, l+1)
    return s

#-------------------------------------------------------------------------------------}

def get_colourlist(x, l):
# Converts colour tree to list of colours for output, note the recursion.
    colourlist = []
    (c, n, sub_boxes) = x
    if n>0: colourlist.append(n2rgb(c,l))
    for box in sub_boxes: colourlist += get_colourlist(box, l+1)
    return colourlist

#-------------------------------------------------------------------------------------}

def loss(x, lo, parent, l):
# Finds node that can be moved up tree with least penalty, note the recursion.
    (c, n, sub_boxes) = x
    z = n>>l
    if l>0 and n>0 and z<lo[0]: lo = [z, l, x, parent]
    for box in sub_boxes: lo = loss(box, lo, x, l+1)
    return lo

#-------------------------------------------------------------------------------------}

def gain(x, hi, l):
# Finds node that can be moved down tree with greatest gain, note the recursion.
    if l==6: return hi
    (c, n, sub_boxes) = x
    z = n>>l
    if z>hi[0]: hi = [z, l, x]
    for box in sub_boxes: hi = gain(box, hi, l+1)
    return hi

#-------------------------------------------------------------------------------------}

def move_down(l, box):
# Move colours down a level
    global coltree, numcols
    (c, n, sub_boxes) = box
# Colour root; sub boxes are coloured 8*c + j.
    cc = c << 3
    threshold = n/8
# Make list of unused subboxes.
    z = [0,1,2,3,4,5,6,7,]
    for sub in sub_boxes: z.remove(sub[0] - cc)
# Get pixels at lower level
    q = p[l+1][cc:cc+8]
    for j in z:
# Don't make small numbers of pixels into new colour.
        if q[j] <= threshold: continue
        newcol = cc + j
# Add entry in list of subboxes and increase count of colours.
        box[2].append([newcol, q[j], []])
        numcols += 1
        box[1] -= q[j]
# If all pixels moved down original colour not used.
    if box[1]==0: numcols -= 1
    return box

#-------------------------------------------------------------------------------------}

def move_up(l, box, parent):
# Moves node up a level
    global coltree, numcols
    (c, n, sub_boxes) = box
    newcol = c >> 3
    i = parent[2].index(box)
    sub=parent[2][i]
# If the parent box had no pixels we create new colour.
    if parent[1]==0: numcols += 1
# Move pixels from box to parent
    parent[1] += n
    parent[2][i][1] = 0
    numcols -= 1
# If there are no sub boxes delete box
    if not box[2]: del parent[2][i]
    return

#-------------------------------------------------------------------------------------}

def my_palette(small):
# Computes optimum colour map using algorithm described in appendix to
# restore2.pdf
    global p, coltree, numcols
    pxls = list(small.getdata())

# Note that the PIL module method .getdata() produces a list of tuples whereas
# the gimp procedure produces a simple list of colour values in order r g b.

# Create lists to contain counts of numbers of pixels of particuler colour.
    p0 = []
    num_levels = 6
    for level in range(0, num_levels + 1):
        p0.append(array('i', [0 for i in range(0, 8**level)]))
# Count pixels in different levels.
    p = p0
    it=iter(pxls)
    try:
        while True:
            (r,g,b)=next(it)
# The function rgb2n converts rgb triplet into integer used to index boxes.
            i = rgb2n(r, g, b)
            p[6][i]     += 1
            p[5][i>>3]  += 1
            p[4][i>>6]  += 1
            p[3][i>>9]  += 1
            p[2][i>>12] += 1
            p[1][i>>15] += 1
            p[0][i>>18] += 1

    except StopIteration: pass

# Construct colour tree.  A node is a tuple (colour number, number of pixels
# of that colour, list of nodes in the next level of the tree).  Colour
# numbers are defined as follows.  Let a  colour value c 0<=c<256
# have binary representation c7 c6 c5 c4 c3 c2 c1 c0;  here c can be r,g or b.
# A level 6 colour number has the binary representation 
# r7 g7 b7 r6 g6 b6 r5 g5 b5 r4 g4 b4 r3 g3 b3 r2 g2 b2.  The corresponding 
# colours at each higher level are obtained by right shifting this 3 places.

# The initial colour tree contains the 64 colours at level 2
    numcols = 0
# level2 is list of boxes at level 2.
    level2 = [[] for i in range(0,8)]
    for i in range(0,8):
        for j in range(0,8):
            c = 8*i+j
            if p[2][c]>0:
                level2[i].append([c, p[2][c], []])
                numcols += 1
# level1 is list of boxes at level 1
    level1 = []
    for i in range(0,8):
        if level2[i]: level1.append([i, 0, level2[i]])
    coltree = [0, 0, level1]

# Set target number of colours.
    col_targ = 256
# Start with a very bad fit
    lastfit = 1e10
# k counts colour moves in case of failure.
    k = 0
    while True:
# If the number of colours is less than required find the box which, if split,
# produces the greatest improvement in the fit.
        if numcols<col_targ:
            best_gain = gain(coltree, [0, None, None], 0)
            if best_gain[0]==0:
                print("Less than 256 distinct colours, impossible to restore")
                return None
            move_down(best_gain[1], best_gain[2])
# note fit before moving colours up the tree in case we need to exit.
        s = colourfit(coltree, 0)
# If the number of colours is too large find the box which, if the colours are
# moved up a level causes the least deterioration in the fit.
        if numcols>=col_targ:
            least_loss = loss(coltree, [1e10, None, None], [coltree], 0)
            move_up(least_loss[1], least_loss[2], least_loss[3])
# If we have the right number of colours exit if fit getting worse.
        if numcols==col_targ:
            nowfit = s
            if nowfit >= lastfit: break
            lastfit = nowfit
# Count moves up and down.
        k = k + 1
# Force exit in exceptional circumstances.
        if k>200: break

# Unpack colour tree into a list of colours and sort these according to 
# their brightness. 
    colours = get_colourlist(coltree, 0)
    colours.sort(key=lambda C1: Y(C1[0], C1[1], C1[2]))

    return list(zip(*colours))

###############################################################################
# End of rountines for getting optimum colours
###############################################################################


# Make look-up table for constructing list index.  Let a colour value 0<=i<256
# have binary representation c7 c6 c5 c4 c3 c2 c1 c0 then
# ctab[i] =  c7 0 0 c6 0 0 c5 0 0 c4 0 0 c3 0 0 c2 0 0 c1 0 0 c0.
ctab=[]
for c in range(0,256):
    i=0
    mask=0x80
    for j in range(0,8):
        i=(i << 2) | (c & mask)
        mask = mask >> 1
    ctab.append(i)

	
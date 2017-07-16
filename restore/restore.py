import os
import sys
import PIL
from PIL import Image, ImageStat


# Code was originally written by Geoff Daniell
# 25/5/17 improvements to code, no change in output.

# The letters RGB are used throughout for referring to colours.
R = 0; G = 1; B = 2

gamma_target = 1.0
sat_choice = 1

# Set parameters here used in colour balance, these seem about optimum.
grey = 50.0; u_off = 2.0; v_off = 2.0

# Define colours of 'ideal' image'
ideal_col = [int(255.0 * (i/255.0)**gamma_target) for i in range(0,256)]

# Make look up table for conversion of RGB to LUV, needs to be bigger 
# than 256 in case search explores here.
expand=[0.0 for c in range(0,360)]
for c in range(0,256):
	C = c/255.0
	if C > 0.04045: C = ((C + 0.055)/1.055)**2.4
	else: C = C/12.92
	expand[c] = 100.0*C

# switch this to use custom colourmap code
if 1:
	def get_colourmap(image):
	# Get the best 256 colours which represent the image.
	# Make a copy for trial restoration.
		copy_small = image.copy()
		copy_small = copy_small.quantize(colors=256, method=0)
		colourmap = copy_small.getpalette()
	# Unpack colourmap and get parameters for extrapolation.
	# See restore2.pdf for details.
		colmap = [0,0,0]; C_hi = [0,0,0]
		for C in [R,G,B]:
			colmap[C] = colourmap[slice(C,768,3)]
	# PIL colour list is in reverse order to gimp order.
			colmap[C] = colmap[C][::-1]
			s0 = sum(colmap[C])
			s1 = sum([i*colmap[C][i] for i in range(0,256)])
			C_hi[C] = int(0.5*(max(colmap[C]) + 3.04e-5*(3.0*s1 - 254.0*s0)))
			C_hi[C] = min([C_hi[C], 255])

		return (colmap, C_hi)
else:
	from colormap import get_colourmap

#-------------------------------------------------------------------------------------}



def RGB2LUV(r,g,b):
# Converts rgb colour values to L* u* v*.
    rr = expand[r]; gg = expand[g]; bb = expand[b]
    X = rr*0.4124 + gg*0.3576 + bb*0.1805
    Y = rr*0.2126 + gg*0.7152 + bb*0.0722
    d = rr*3.6593 + gg*11.4432 + bb*4.115 + 1e-10
    U = 4.0*X/d
    V = 9.0*Y/d 
    Y = Y/100.0
    if Y > 0.008856: Y = Y**0.333333
    else:            Y = 7.787*Y + 0.1379
    Lstar = 116.0*Y - 16.0
    ustar = 13.0*Lstar*(U - 0.1978398)
    vstar = 13.0*Lstar*(V - 0.4683363)
    return (Lstar, ustar, vstar)

#-------------------------------------------------------------------------------------}


def simplex(x0, scale, F, eps, debug):
# Minimises function F in n dimensions using Nelder-Meade method starting
# from vector x0.  Parameter debug is not used.
# Exits if has found set of points with (|high| - |low|)/(|high| + |low|) < eps
# or number of function evaluations exceeds 5000.
# On exit returns coordinates of minimum and True or False, depending on 
# number of function evaluations used.
    ok = False
# Get number of dimensions.
    n = len(x0)
# Set up initial simplex.
    p = [x0[:] for i in range(0,n+1)]
    for i in range(0,n): p[i][i] += scale[i]
    psum = [sum([p[i][j] for i in range(0,n+1)]) for j in range(0,n)]
    nfunc = 0
# Get function value at vertices.
    y = [F(p[i]) for i in range(0,n+1)]
    while True:
# Get highest.
        hi = y.index(max(y))
# Set this value very low and get next highest.
        (y[hi], ysave) = (-1e10, y[hi])
        next_hi = y.index(max(y))
        y[hi] = ysave
# Get lowest.
        lo = y.index(min(y))
# Test for convergence.
        if 2.0*abs(y[hi] - y[lo])/(abs(y[hi]) + abs(y[lo])) < eps:
            ok = True
            break
# Exit if failed to converge.
        if nfunc>5000: break

        nfunc += 2
        (ynew, p, y, psum) = trial(p, y, psum, n, F, hi, -1.0)
# If new point better try going further.
        if ynew <= y[lo]:
            (ynew, p, y, psum) = trial(p, y, psum, n, F, hi, 2.0)
# If the new point is worse than the next highest ...
        elif ynew >= y[next_hi]:
            ysave = y[hi]
            (ynew, p, y, psum) = trial(p, y, psum, n, F, hi, 0.5)
# If getting nowhere shrink the simplex.
            if ynew >= ysave:
# Loop over vertices keeping the lowest point unchanged.
                for i in range(0,n+1):
                      if i==lo: continue
                      pnew = [0.5*(p[i][j] + p[lo][j]) for j in range(0,n)]
                      p[i] = pnew
                      y[i] = F(pnew)
            nfunc += n
            psum = [sum([p[i][j] for i in range(0,n+1)]) for j in range(0,n)]
        else: nfunc -= 1
    return (p[lo], ok)

def trial(p, y, psum, n, F, hi, dist):
# Compute point pnew along line from p[hi] to centroid excluding p[h1].
    a = (1.0 - dist)/n
    b = a - dist
    pnew = [a*psum[j] - b*p[hi][j] for j in range(0,n)]
    ynew = F(pnew)
# If improvement accept and adjust psum.
    if ynew < y[hi]:
        y[hi] = ynew
        psum = [psum[j] + (pnew[j] - p[hi][j]) for j in range(0,n)]
        p[hi] = pnew
    return (ynew, p, y, psum)

#-------------------------------------------------------------------------------------}

def first_restore():
# Optimises lambda and sigma separately for each colour channel.
    global CH
    CH = R; scale = [0.02, 0.02]; x0 = [1.0, 1.0]
    ((lamr, sigr), ok_r) = simplex(x0, scale, ideal_colour, 1e-4, True)
    CH = G; scale = [0.02, 0.02]; x0 = [1.0, 1.0]
    ((lamg, sigg), ok_g) = simplex(x0, scale, ideal_colour, 1e-4, True)
    CH = B; scale = [0.02, 0.02]; x0 = [1.0, 1.0]
    ((lamb, sigb), ok_b) = simplex(x0, scale, ideal_colour, 1e-4, True)
    if not (ok_r and ok_g and ok_b):
        pdb.gimp_message("The program has failed to obtain a satisfactory"
                          "restoration; the result shown may be poor.")
    return (lamr, lamg, lamb, sigr, sigg, sigb)

#-------------------------------------------------------------------------------------}

def ideal_colour(p):
# Calculates measure of misfit between actual colours and ideal colours.
# This is to be minimised in the first stage of restoration.
    lamc, sigc = p
    measure = 0.0
    for i in range(0,256):
        c = int(255 * sigc * (colmap[CH][i]/255.0)**lamc)
        measure += (c - ideal_col[i])**2
    return measure

#-------------------------------------------------------------------------------------}

def colour_balance(p):
# Calculates weighted average distance in U* v* space from u_off, v_off.
# This is minimised in the second stage of restoration
    lamr, lamg, lamb, sigr, sigg, sigb = p
    usum = 0.0; vsum = 0.0; wsum = 0.0
    for i in range(0,256):
# Compute the colour as modified by the trial restoration parameters.
        r = int(255 * sigr * (colmap[R][i]/255.0)**lamr)
        g = int(255 * sigg * (colmap[G][i]/255.0)**lamg)
        b = int(255 * sigb * (colmap[B][i]/255.0)**lamb)
        (Lstar, ustar, vstar) = RGB2LUV(r,g,b)
        s = ustar*ustar + vstar*vstar
# Weight so that only pale colours are considered.
        w = grey/(grey + s)
        usum = usum + w*ustar
        vsum = vsum + w*vstar
        wsum = wsum + w
    dist = (usum/wsum - u_off)**2 + (vsum/wsum - v_off)**2
    return dist

#-------------------------------------------------------------------------------------}

def levels_params(Lambda, Sigma, c_hi):
# Converts restoration parameters Lambda and Sigma to those used in
# the gimp levels command.
    alpha = [0,0,0]
    s = [0,0,0]
    for C in [R,G,B]: s[C] = Sigma[C] * (c_hi[C]/255.0)**Lambda[C]
    smax = max(s)
    for C in [R,G,B]:
        alpha[C] = 1.0/Lambda[C]
        s[C] = int(255.0*(s[C]/smax))
    return (alpha, c_hi, s)

#-------------------------------------------------------------------------------------}

def adjust_levels(image, hi_in, gamma, hi_out):
# Function is a replacement for the gimp levels command.
# Split image into R ,G and B images and process separately.
    RGB = list(image.split())
    for C in [R,G,B]:
        alpha = 1.0/float(gamma[C])
        float_hi_in = float(hi_in[C])
        RGB[C]=RGB[C].point(lambda I: int(hi_out[C]*(I/float_hi_in)**alpha+0.5))
# Return merged image.
    return Image.merge("RGB", tuple(RGB))

###############################################################################

def restore(im):
    global colmap
# Create a small image to speed determination of resoration parameters.
    (width, height)=im.size
    small_image=im.resize((int(width/8), int(height/8)), resample=PIL.Image.NEAREST)
# Get colourmap of small image, using local code.
    cols=get_colourmap(small_image)
# Exit if failed to get colourmap.
    if cols==None: return None

#    if debug: print cols
    (colmap, C_hi)=cols
# Get first estimate of restoration parameters.
    (lamr, lamg, lamb, sigr, sigg, sigb) = first_restore()
    Lambda1 = [lamr, lamg, lamb]; Sigma1 = [sigr, sigg, sigb]

#    if debug:
#        print C_hi
#        print "Lambda1, Sigma1", Lambda1, Sigma1
#
# Convert resoration parameters Lambda1 and Sigma1 to parameters for
# gimp levels command.
    (alpha1, m1, s1) = levels_params(Lambda1, Sigma1, C_hi)
# Restore the small image using replacement for gimp levels command and
# get its colourmap.
    restored_small = adjust_levels(small_image, m1, alpha1, s1)
#    if debug: print "m1, alpha1, s1", m1, alpha1, s1
    (colmap, junk) = get_colourmap(restored_small)

# Do second stage of restoration to adjust colour balance.
    scale = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
    x0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    fit = simplex(x0, scale, colour_balance, 1e-4, True)[0]
    lamr, lamg, lamb, sigr, sigg, sigb, = fit
    Lambda2 = [lamr, lamg, lamb]; Sigma2 = [sigr, sigg, sigb]

#    if debug: print "Lambda2, Sigma2", Lambda2, Sigma2

# Combine the parameters for both stages of restoration.
    Lambda3 = [0,0,0]; Sigma3 = [0,0,0]
    for C in [R,G,B]:
        Sigma3[C] = Sigma2[C] * Sigma1[C]**Lambda2[C]
        Lambda3[C] = Lambda2[C] * Lambda1[C]

# Get parameters for gimp levels command.
    (alpha2, m2, s2) = levels_params(Lambda3, Sigma3, C_hi)

#    if debug: print "m2, alpha2, s2", m2, alpha2, s2

# Restore main full size image.
    restored_image = adjust_levels(im, m2, alpha2, s2)

# Generate a more saturated option if requested.
    if sat_choice:
# Convert image to HSV format and split into separate images.
        HSVimage = restored_image.convert("HSV")
        HSV=list(HSVimage.split())
# Get statistics of the S channel to compute new saturation.
        stats = ImageStat.Stat(HSV[1])
        mean = stats.mean[0]
        std_dev = stats.stddev[0]
# Compute an estimate of high saturation values and factor by which to scale.
        maxsat = mean + 2.0*std_dev
        fac = 1.0/min(1.0, 1.0/min(1.5, 150.0/maxsat))
# Increase the values in the saturation channel, merge HSV and convert to RGB.
        HSV[1]=HSV[1].point(lambda I: int(fac*I+0.5))
        more_saturated = Image.merge("HSV", tuple(HSV))
        more_saturated = more_saturated.convert("RGB")
        return more_saturated
    else: return restored_image


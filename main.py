import os
import sys
import PIL
from PIL import Image, ImageStat
from restore import restore

def main():
	# Set default parameters.
	debug = False
	directory = os.getcwd()

	# Parse the command line parameters.
	for param in sys.argv:
		Dir = param.find("dir=")
		if Dir>=0: directory = param[Dir+4:]
		gam = param.find("Light-Dark=")
		if gam>=0: gammatarg = float(param[gam+11:])
		sat = param.find("saturate=")
		if sat>=0: sat_choice = eval(param[sat+9:])

	# if debug: print directory, gammatarg, sat_choice

	# Save the current directory.
	savedir = directory
	# Make list of files to process.
	os.chdir(directory)
	filelist = os.listdir(directory)
	jpegs = []; JPEGS = []; tiffs = []; TIFFS = []
	for File in filelist:
		if File.find(".jpg")>0:  jpegs.append(File)
		if File.find(".JPG")>0:  JPGES.append(File)
		if File.find(".tiff")>0: tiffs.append(File)
		if File.find(".TIFF")>0: TIFFS.append(File)

	# In windows the file searching is NOT case sensitive, so merge.
	if JPEGS!=jpegs: jpegs += JPEGS
	if TIFFS!=tiffs: tiffs += TIFFS

	# Loop over the photos to be processed.
	for photo in jpegs+tiffs:
	# Strip off directory name and .jpg to get file name.
		photoname = os.path.split(photo)[1]
		print(photoname)
	# Open photo.
		im = Image.open(photoname)
	# Restore the image.
		restored_image = restore(im)
	# Save file in subdirectory "restored"
		newfilename = os.path.join(directory, "restored", photoname)
		restored_image.save(newfilename, icc_profile=im.info.get('icc_profile'))

	# Return to saved directory at end.
	os.chdir(savedir)
	
if __name__ == '__main__':
	main()


#parse ibug xml for landmarks needed for masking, exclude landmarks not needed for masking
#used this tutorial for parsing https://www.pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/

import argparse
import re

# manage input and output files
arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("-i", "--input", required=True,
	help="input file to parse")
arg_parse.add_argument("-o", "--output", required=True,
	help="output file produced")
args = vars(arg_parse.parse_args())


# landmarks from ibug that we want to use for determining if a face is present
CHIN=list(range(0,17))
NOSE=list(range(29, 36))
MARKS=set(CHIN+NOSE)

# to easily parse chin and nose locations from original xml
# use regex to determine if there is a 'part' element on any given line of xml
PART = re.compile("part name='[0-9]+'")
# load the contents of the original XML file and open the output file
# for writing
print("[INFO] parsing data split XML file...")
rows = open(args["input"]).read().strip().split("\n")
output = open(args["output"], "w")

# loop over xml and parse out needed parts for chin an nose
for row in rows:
	# check to see if the current line has the (x, y)-coordinates for
	# the facial landmarks we are interested in
	parts = re.findall(PART, row)
	# if there no parts found just write out no processing needed we just need to duplicate what is in the original file
	if len(parts) == 0:
		output.write("{}\n".format(row))
	# otherwise, there is annotation information that we must process
	else:
		# parse out the name of the attribute from the row
		attr = "name='"
		i = row.find(attr)
		j = row.find("'", i + len(attr) + 1)
		name = int(row[i + len(attr):j])
		# if the facial landmark name exists within the range of our
		# indexes, write it to our output file
		if name in MARKS:
			output.write("{}\n".format(row))
# close the output file
output.close()
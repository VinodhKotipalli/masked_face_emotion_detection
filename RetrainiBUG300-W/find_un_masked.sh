#!/bin/sh

SCRIPT_DIR=`cd $(dirname "$0") && pwd`

DIR_O="$1"
DIR_M="$2"
DEST="$1_unmask"

mkdir -p "$SCRIPT_DIR/$DEST"

for i in $( find "$SCRIPT_DIR/$DIR_O" -type f -exec basename {} \; ); do
   file_name=$( echo $i | cut -f1 -d"." )
   masked=$( ls "$SCRIPT_DIR/$DIR_M" | grep -c "$file_name" )
   if [ $masked -eq 0 ]; then
	cp "$SCRIPT_DIR/$DIR_O/$i" "$SCRIPT_DIR/$DEST"
   fi
done

#!/bin/bash
FILES=completed_images/*
save_directory="completed_images_png"

if [ ! -z "$1" ]
  then
    echo "Processing $1 file..."
    f="$1"
    basename "$f"
    filename="$(basename -- $f)"
    filename=$filename".png"
    inkscape -z -e $save_directory/$filename -d 1000 $f
fi

if [ $# -eq 0 ]
  then
    for f in $FILES
    do
        echo "Processing $f file..."
        # Getting the extension
        basename "$f"
        filename="$(basename -- $f)"
        filename=$filename".png"
        inkscape -z -e $save_directory/$filename -d 1000 $f
        echo "Finished"
    done
fi


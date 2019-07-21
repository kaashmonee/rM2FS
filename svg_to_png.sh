#!/bin/bash
FILES=completed_images/*
save_directory="completed_images_png"

for f in $FILES
do
    echo "Processing $f file..."
    # Getting the extension
    basename "$f"
    filename="$(basename -- $f)"
    filename=$filename".png"
    inkscape -z -e $save_directory/$filename -w 1000 -h 1000 $f
    echo "Finished"
done


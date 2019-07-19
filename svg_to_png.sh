#!/bin/bash
FILES=completed_images/*
for f in $FILES
do
    echo "Processing $f file..."
    filename="$f.png"
    inkscape -z -e completed_images_png/$filename -w 1000 -h 1000 $f
    echo "Finished"
done


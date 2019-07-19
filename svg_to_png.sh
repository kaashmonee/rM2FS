#!/bin/bash
FILES=completed_images/*
for f in $FILES
do
    echo "Processing $f file..."
    filename="$f.png"
    inkscape -z -e completed_images_png/$filename
    echo "Finished"
done


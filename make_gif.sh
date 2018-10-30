files="predicted/frame20.png"
for i in $(seq 1 19); do
	files="predicted/frame${i}.png ${files}"
done
convert -delay 30 -loop 0 $files "pupils.gif"

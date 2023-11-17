FNAME=${1}
WIDTH=${2}

ffmpeg \
    -i ${FNAME} \
    -vf "fps=10,scale=${WIDTH}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
    -loop 0 \
    sway.gif

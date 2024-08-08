echo $SHELL
if [ ! -d "build" ]; then
    mkdir build
fi
if [ -d "build" ]; then
    echo "rm -rf build"
    rm -rf build
    mkdir build
fi

cmake -B build && cmake --build build -j16
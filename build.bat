@echo off
echo Building NeuroForge for Windows...
echo.

REM Check if MinGW is available
where gcc >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: GCC not found. Please install MinGW-w64.
    echo Download from: https://www.mingw-w64.org/downloads/
    pause
    exit /b 1
)

REM Create directories
if not exist obj mkdir obj
if not exist bin mkdir bin

REM Build the library
echo Building library...
gcc -Wall -Wextra -O3 -fopenmp -c src/matrix.c -o obj/matrix.o
if %errorlevel% neq 0 (
    echo Error building matrix.o
    pause
    exit /b 1
)

gcc -Wall -Wextra -O3 -fopenmp -c src/network.c -o obj/network.o
if %errorlevel% neq 0 (
    echo Error building network.o
    pause
    exit /b 1
)

REM Build layers
echo Building layers...
gcc -Wall -Wextra -O3 -fopenmp -c src/layers/dense.c -o obj/dense.o
gcc -Wall -Wextra -O3 -fopenmp -c src/layers/conv2d.c -o obj/conv2d.o
gcc -Wall -Wextra -O3 -fopenmp -c src/layers/rnn.c -o obj/rnn.o
gcc -Wall -Wextra -O3 -fopenmp -c src/layers/attention.c -o obj/attention.o
gcc -Wall -Wextra -O3 -fopenmp -c src/layers/dropout.c -o obj/dropout.o

REM Build optimizers
echo Building optimizers...
gcc -Wall -Wextra -O3 -fopenmp -c src/optimizers/sgd.c -o obj/sgd.o
gcc -Wall -Wextra -O3 -fopenmp -c src/optimizers/adam.c -o obj/adam.o
gcc -Wall -Wextra -O3 -fopenmp -c src/optimizers/rmsprop.c -o obj/rmsprop.o

REM Build activations
echo Building activations...
gcc -Wall -Wextra -O3 -fopenmp -c src/activations/activation.c -o obj/activation.o

REM Build serialization
echo Building serialization...
gcc -Wall -Wextra -O3 -fopenmp -c src/serialization.c -o obj/serialization.o

REM Create static library
echo Creating library...
ar rcs bin/libneuroforge.a obj/*.o

REM Build examples
echo Building examples...
gcc -Wall -Wextra -O3 -fopenmp examples/mnist.c -o bin/mnist.exe bin/libneuroforge.a -lm -fopenmp
gcc -Wall -Wextra -O3 -fopenmp examples/xor.c -o bin/xor.exe bin/libneuroforge.a -lm -fopenmp
gcc -Wall -Wextra -O3 -fopenmp examples/transformer.c -o bin/transformer.exe bin/libneuroforge.a -lm -fopenmp

REM Build tests
echo Building tests...
gcc -Wall -Wextra -O3 -fopenmp tests/test_layers.c -o bin/test_layers.exe bin/libneuroforge.a -lm -fopenmp
gcc -Wall -Wextra -O3 -fopenmp tests/test_optimizers.c -o bin/test_optimizers.exe bin/libneuroforge.a -lm -fopenmp
gcc -Wall -Wextra -O3 -fopenmp tests/text_matrix.c -o bin/test_matrix.exe bin/libneuroforge.a -lm -fopenmp

echo.
echo Build completed successfully!
echo.
echo Files created:
echo   - bin/libneuroforge.a (static library)
echo   - bin/mnist.exe (MNIST example)
echo   - bin/xor.exe (XOR example)
echo   - bin/transformer.exe (Transformer example)
echo   - bin/test_*.exe (test programs)
echo.
echo To run examples:
echo   bin\mnist.exe
echo   bin\xor.exe
echo   bin\transformer.exe
echo.
echo To run tests:
echo   bin\test_matrix.exe
echo   bin\test_layers.exe
echo   bin\test_optimizers.exe
echo.
pause

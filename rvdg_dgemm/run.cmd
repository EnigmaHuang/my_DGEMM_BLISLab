echo off

set EXENAME=TestMM.exe
set CFLAGS=-O2 -fopenmp -march=native -m64
set LIBPATH=-I"D:\Dev-Cpp\MinGW64\include" -I"D:\Dev-Cpp\MinGW64\x86_64-w64-mingw32\include" -I"D:\Dev-Cpp\MinGW64\lib\gcc\x86_64-w64-mingw32\4.9.2\include\c++"  -L"D:\Dev-Cpp\MinGW64\lib" -L"D:\Dev-Cpp\MinGW64\x86_64-w64-mingw32\lib"

set NEW=MMult_4x4_15
set OLD=MMult_4x4_9

del *.o %EXENAME%
del output_%NEW%.m

D:\Dev-Cpp\MinGW64\bin\g++ %CFLAGS% %LIBPATH% -c Ref.cc 
D:\Dev-Cpp\MinGW64\bin\g++ %CFLAGS% %LIBPATH% -c %NEW%.cc 
D:\Dev-Cpp\MinGW64\bin\g++ %CFLAGS% %LIBPATH% -c TestMM.cc
D:\Dev-Cpp\MinGW64\bin\g++ %CFLAGS% %LIBPATH% -o %EXENAME% Ref.o %NEW%.o TestMM.o

echo Compile Finished. Test start...

%EXENAME% > output_%NEW%.m
copy output_%OLD%.m output_old.m
copy output_%NEW%.m output_new.m

type output_%NEW%.m
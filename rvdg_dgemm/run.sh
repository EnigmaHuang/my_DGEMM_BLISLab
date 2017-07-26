#!/bin/bash

EXENAME=TestMM.x

OLD=MMult_4x4_15
NEW=MMult_4x4_15

CFLAGS="-O2 -Wall -march=native -fopenmp"

rm *.o $EXENAME
rm output_$NEW.m
g++ $CFLAGS -c Ref.cc 
g++ $CFLAGS -c $NEW.cc 
g++ $CFLAGS -c TestMM.cc
g++ $CFLAGS -o $EXENAME Ref.o $NEW.o TestMM.o

echo "Test start..."

./$EXENAME >> output_$NEW.m
cp output_$OLD.m output_old.m
cp output_$NEW.m output_new.m

cat output_new.m

echo "Test end."

#!/bin/sh

for args in `seq 0 9`
do
    grep GB ${args}*.csv
done

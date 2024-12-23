#!/bin/bash

start=1  # 시작 번호 (test1)
end=23   # 끝 번호 (test20)

for ((test=$start; test<=$end; test++)); do
    for i in {001..007}; do
        cp test1_$i.pdb test${test}_$i.pdb
    done
done

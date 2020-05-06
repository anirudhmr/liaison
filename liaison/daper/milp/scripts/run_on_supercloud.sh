#!/bin/bash
# Run on supercloud.
FILE=$1

# read SLURM_NODEID line from the
(( x = ${SLURM_NODEID} + 1 ))

eval `sed "${x}q;d" < $FILE`

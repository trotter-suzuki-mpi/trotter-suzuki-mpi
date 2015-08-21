#!/bin/sh
#@ wall_clock_limit = 00:04:00
#@ initialdir = .
#@ error = trotter%j.err
#@ output = trotter%j.out
#@ total_tasks = 1
#@ cpus_per_task = 4
#@ node_usage = not_shared

export NX_DISABLECUDA=yes 
export OMP_NUM_THREADS=4

./build/trottertest

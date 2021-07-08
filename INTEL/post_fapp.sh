#!/bin/sh
#!/bin/bash
#SBATCH --partition=sahara
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J fapp

module load fujitsu
export FLIB_FASTOMP=TRUE

export OMP_NUM_THREADS=48

fapppx -A -d ./rep_s1  -Icpupa,nompi -tcsv -o pa_s1.csv
fapppx -A -d ./rep_s2  -Icpupa,nompi -tcsv -o pa_s2.csv
fapppx -A -d ./rep_s3  -Icpupa,nompi -tcsv -o pa_s3.csv
fapppx -A -d ./rep_s4  -Icpupa,nompi -tcsv -o pa_s4.csv
fapppx -A -d ./rep_s5  -Icpupa,nompi -tcsv -o pa_s5.csv
fapppx -A -d ./rep_s6  -Icpupa,nompi -tcsv -o pa_s6.csv
fapppx -A -d ./rep_s7  -Icpupa,nompi -tcsv -o pa_s7.csv
fapppx -A -d ./rep_s8  -Icpupa,nompi -tcsv -o pa_s8.csv
fapppx -A -d ./rep_s9  -Icpupa,nompi -tcsv -o pa_s9.csv
fapppx -A -d ./rep_s10 -Icpupa,nompi -tcsv -o pa_s10.csv
fapppx -A -d ./rep_s11 -Icpupa,nompi -tcsv -o pa_s11.csv
fapppx -A -d ./rep_s12 -Icpupa,nompi -tcsv -o pa_s12.csv
fapppx -A -d ./rep_s13 -Icpupa,nompi -tcsv -o pa_s13.csv
fapppx -A -d ./rep_s14 -Icpupa,nompi -tcsv -o pa_s14.csv
fapppx -A -d ./rep_s15 -Icpupa,nompi -tcsv -o pa_s15.csv
fapppx -A -d ./rep_s16 -Icpupa,nompi -tcsv -o pa_s16.csv
fapppx -A -d ./rep_s17 -Icpupa,nompi -tcsv -o pa_s17.csv

fapppx -A -d ./rep_d1  -Icpupa,nompi -tcsv -o pa_d1.csv
fapppx -A -d ./rep_d2  -Icpupa,nompi -tcsv -o pa_d2.csv
fapppx -A -d ./rep_d3  -Icpupa,nompi -tcsv -o pa_d3.csv
fapppx -A -d ./rep_d4  -Icpupa,nompi -tcsv -o pa_d4.csv
fapppx -A -d ./rep_d5  -Icpupa,nompi -tcsv -o pa_d5.csv
fapppx -A -d ./rep_d6  -Icpupa,nompi -tcsv -o pa_d6.csv
fapppx -A -d ./rep_d7  -Icpupa,nompi -tcsv -o pa_d7.csv
fapppx -A -d ./rep_d8  -Icpupa,nompi -tcsv -o pa_d8.csv
fapppx -A -d ./rep_d9  -Icpupa,nompi -tcsv -o pa_d9.csv
fapppx -A -d ./rep_d10 -Icpupa,nompi -tcsv -o pa_d10.csv
fapppx -A -d ./rep_d11 -Icpupa,nompi -tcsv -o pa_d11.csv
fapppx -A -d ./rep_d12 -Icpupa,nompi -tcsv -o pa_d12.csv
fapppx -A -d ./rep_d13 -Icpupa,nompi -tcsv -o pa_d13.csv
fapppx -A -d ./rep_d14 -Icpupa,nompi -tcsv -o pa_d14.csv
fapppx -A -d ./rep_d15 -Icpupa,nompi -tcsv -o pa_d15.csv
fapppx -A -d ./rep_d16 -Icpupa,nompi -tcsv -o pa_d16.csv
fapppx -A -d ./rep_d17 -Icpupa,nompi -tcsv -o pa_d17.csv

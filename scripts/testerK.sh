#!/bin/bash 

K=7
	L=7
	while [  $K -le 20 ]; do
         echo "Running crossValidation with K="$K "L="$L "threads="$1 "cross="$2;
        ./knn train.csv test.csv crossValidation.txt $K $L $1 $2
		let K=K+1 
	done
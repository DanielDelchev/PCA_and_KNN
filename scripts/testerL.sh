#!/bin/bash 

K=11
	L=5
	while [  $L -le 7 ]; do
         echo "Running crossValidation with K="$K "L="$L "threads="$1 "cross="$2;
        ./knn train.csv test.csv crossValidation.txt $K $L $1 $2
		let L=L+1 
	done
done
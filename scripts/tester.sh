#!/bin/bash 

K=1
while [  $K -le 20 ]; do

	L=2
	while [  $L -le 7 ]; do
         echo "Running crossValidation with K="$K "L="$L "threads="$1 "cross="$2;
        ./knn train.csv test.csv crossValidation.txt $K $L $1 $2
		let L=L+1 
	done

	let K=K+1
done
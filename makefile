CC=g++
CFLAGS=-Wall -std=c++11 -O3 -pthread
INCLUDE = -I external/TNT -I external/JAMA
LINK = -L .

all:pca knn

pca:pca.cpp
	${CC} pca.cpp -o pca ${CFLAGS} ${INCLUDE} ${LINK}

knn:knn.cpp
	${CC} knn.cpp -o knn ${CFLAGS}
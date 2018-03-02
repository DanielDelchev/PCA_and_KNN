#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <unordered_map>
#include <chrono>
#include <ratio>

static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
static std::default_random_engine generator (seed);
static std::uniform_int_distribution<int> toss(0,1);

struct Model {
    Model (size_t _pixelCount = 784):pixelCount(_pixelCount){}
    Model(std::vector<std::vector<double>> _features, std::vector<int> _labels, std::vector<size_t> _keys, size_t _pixelCount=784): \
        features(_features),labels(_labels),keys(_keys),pixelCount(_pixelCount){}
    void train(std::string filename = "train.csv");

    std::vector<int> predict(std::string filename = "test.csv", size_t Ldist = 3,size_t threadCount = 1, size_t K = 11)const;

    double crossValidate(size_t fold = 10,size_t Ldist = 3,size_t threadCount = 1, size_t K = 11)const;

    std::vector<std::vector<double>> features;
    std::vector<int> labels;
    std::vector<size_t> keys;

    size_t pixelCount;
};

void Model::train(std::string filename){

    features.clear();
    labels.clear();
    keys.clear();

    //try to open filestream
    std::fstream fileStream(filename.c_str(),std::ios::in);
    if (!fileStream.is_open()){
        perror("Could not open file!\n");
        exit (1);
    }

    std::string line;
    int currentKey = -1;

    //skip headings
    getline(fileStream,line);

    //get next line
    while (getline(fileStream,line))
    {
        std::vector<double> pixels;
        pixels.reserve(pixelCount);
        ++currentKey;

        //split line by " " , "," "tab"
        char* chunk = strtok(const_cast<char*>(line.c_str())," \t,");
        // get Label
        labels.push_back(atof(chunk));
        //get Features
        size_t counter = 0;
        if (pixelCount>0){
            while ( (chunk = strtok(nullptr," \t,"))!=nullptr && counter<pixelCount){
                pixels.push_back(atof(chunk));
                counter++;
            }
        }
        else{
            while ( (chunk = strtok(nullptr," \t,"))!=nullptr){
                pixels.push_back(atof(chunk));
            }
        }
        //store them
        features.push_back(pixels);
        keys.push_back(currentKey);
    }
    //close filestream

    pixelCount = features[0].size();

    fileStream.close();
}

template<typename T>
static double Lnorm (const std::vector<T>& one, const std::vector<T>& other, int Ldist){
    if (one.size() != other.size()){
        perror("vectors of different lengths passed to Lnorm!\n");
        std::cout<<one.size()<<" "<<other.size()<<std::endl;
        exit(1);
    }
    double sum = 0;
    size_t length = one.size();
    for (size_t index=0; index<length;index++){
        sum += pow(fabs((one[index])-(other[index])),Ldist);
    }

    return pow(sum,(1.0/Ldist));

}



static void predictorWeighted(const Model* const model, const std::vector<size_t> predictKeys,const std::vector<std::vector<double>>& predictFeatures,std::vector<int>& predictLabels, size_t Ldist){

    int prediction = 0;
    size_t patronsCount = model->features.size();
    std::vector<double> distances(patronsCount,0);
    const std::vector<std::vector<double>>& patrons = model->features;
    const std::vector<int>& labels = model->labels;

    const int AMPLIFIER = 10;
    for (const auto& key: predictKeys){
        const std::vector<double>& current = predictFeatures[key];
        for (size_t patronIndex=0;patronIndex<patronsCount;patronIndex++){
            distances[patronIndex] = Lnorm(current,patrons[patronIndex],Ldist);
        }
        std::unordered_map<int,double> poll;
        for (size_t distIndex = 0; distIndex<patronsCount;distIndex++){
            int label = labels[distIndex];

            if (poll.find(label) == poll.end()){
                poll[label] = pow((1/distances[distIndex]),AMPLIFIER);
            }
            else{
                poll[label] += pow((1/distances[distIndex]),AMPLIFIER);
            }
        }

        const int EXP = 10*AMPLIFIER;
        const double EPS = pow(10,-EXP);
        double maxWeight = 0;

        for (const auto& iter: poll){
            if (iter.second - maxWeight > EPS){
                prediction = iter.first;
                maxWeight = iter.second;
            }
            else if( fabs(iter.second-maxWeight)<EPS){
                if(toss(generator)){
                    prediction = iter.first;
                    maxWeight = iter.second;
                }
            }
        }

        predictLabels[key] = prediction;
    }
}


static void predictor(const Model* const model, const std::vector<size_t> predictKeys,const std::vector<std::vector<double>>& predictFeatures,std::vector<int>& predictLabels, size_t Ldist,size_t K){

    int prediction = 0;
    size_t patronsCount = model->features.size();
    std::vector<std::pair<int,double>> distances(patronsCount,{0,0});
    const std::vector<std::vector<double>>& patrons = model->features;
    const std::vector<int>& labels = model->labels;

    const int EXP = 40;
    const int EPS = pow(10,-EXP);
    for (const auto& key: predictKeys){
        const std::vector<double>& current = predictFeatures[key];
        for (size_t patronIndex=0;patronIndex<patronsCount;patronIndex++){
            distances[patronIndex] = {patronIndex,Lnorm(current,patrons[patronIndex],Ldist)};
        }

        std::function< bool(const std::pair<int,double>&, const std::pair<int,double>& )> CMP = [&](const std::pair<int,double>& one, const std::pair<int,double>& other){
            return (other.second - one.second > EPS);
        };

        sort(distances.begin(),distances.end(),CMP);


        std::unordered_map<int,int> poll;
        for (size_t distIndex = 0; distIndex<K;distIndex++){
            int label = labels[distances[distIndex].first];

            if (poll.find(label) == poll.end()){
                poll[label] = 1;
            }
            else{
                poll[label]++ ;
            }
        }

         int counter = 0;
         for (const auto& iter: poll){
            if (iter.second > counter){
                prediction = iter.first;
                counter = iter.second;
            }
            else if(iter.second == counter){
                if(toss(generator)){
                    prediction = iter.first;
                    counter = iter.second;
                }
            }
        }
        predictLabels[key] = prediction;
    }
}


std::vector<int> Model::predict(std::string filename, size_t Ldist,size_t threadCount, size_t K)const{


    std::vector<std::vector<double>> predictFeatures;
    std::vector<size_t> predictKeys;

    //try to open filestream
    std::fstream fileStream(filename.c_str(),std::ios::in);
    if (!fileStream.is_open()){
        perror("Could not open file!\n");
        exit (1);
    }

    std::string line;
    int currentKey = -1;

    //skip headings
    getline(fileStream,line);

    //get next line
    while (getline(fileStream,line))
    {
        std::vector<double> pixels;
        pixels.reserve(pixelCount);
        ++currentKey;

        //split line by " " , "," "tab"
        char* chunk = strtok(const_cast<char*>(line.c_str())," \t,");
        //get Features
        size_t counter = 0;
        do{
            pixels.push_back(atof(chunk));
            counter++;
        }while ( (chunk = strtok(nullptr," \t,"))!=nullptr && counter<pixelCount );

        //store them
        predictFeatures.push_back(pixels);
        predictKeys.push_back(currentKey);
    }
    //close filestream
    fileStream.close();

    std::vector<int> predictLabels(currentKey+1,0);

    shuffle(predictKeys.begin(),predictKeys.end(),generator);
    std::vector<std::thread> threads;
    threads.reserve(threadCount);

    size_t keysCount = predictKeys.size();
    size_t fraction = keysCount / threadCount;

    auto itEnd = predictKeys.begin();
    for (size_t counter = 0; counter<threadCount-1;counter++){
        auto itStart = predictKeys.begin() + fraction*counter;
        itEnd = itStart + fraction;
        std::vector<size_t> part (itStart,itEnd);
        if(K<=0){
            threads.push_back(std::thread(predictorWeighted,this,std::move(part),std::ref(predictFeatures),std::ref(predictLabels),Ldist));
        }
        else {
            threads.push_back(std::thread(predictor,this,std::move(part),std::ref(predictFeatures),std::ref(predictLabels),Ldist,K));
        }
    }

     std::vector<size_t> part (itEnd,predictKeys.end());
    if(K<=0){
        threads.push_back(std::thread(predictorWeighted,this,std::move(part),std::ref(predictFeatures),std::ref(predictLabels),Ldist));
    }
    else {
        threads.push_back(std::thread(predictor,this,std::move(part),std::ref(predictFeatures),std::ref(predictLabels),Ldist,K));
    }

    for (auto& p : threads){
        p.join();
    }

    return predictLabels;
}


double Model::crossValidate(size_t fold, size_t Ldist, size_t threadCount, size_t K)const{
    size_t length = this->features.size();
    if (length == 0){
        perror("Train model before cross validating!\n");
        return 0;
    }


    //split data into number of folds;
    int fraction = length / fold;
    std::vector<std::vector<std::vector<double> > > featuresFolds;
    std::vector<std::vector<int>> labelsFolds;
    std::vector<double> accuracy;

    std::vector<size_t> keysCopy(keys);

    shuffle(keysCopy.begin(),keysCopy.end(),generator);

    for (size_t counter=0;counter<fold-1;counter++){
        size_t from = counter*fraction;
        size_t to = from + fraction;
        std::vector<std::vector<double>> featuresToAdd;
        std::vector<int> labelsToAdd;
        featuresToAdd.reserve(fraction);
        labelsToAdd.reserve(fraction);
        for (size_t keyIndex=from;keyIndex<to;keyIndex++){
            featuresToAdd.push_back(features[keysCopy[keyIndex]]);
            labelsToAdd.push_back(labels[keysCopy[keyIndex]]);
        }
        featuresFolds.push_back(featuresToAdd);
        labelsFolds.push_back(labelsToAdd);
    }

    size_t from = (fold-1)*fraction;
    size_t to = length;
    std::vector<std::vector<double>> featuresToAdd;
    std::vector<int> labelsToAdd;
    featuresToAdd.reserve(fraction);
    labelsToAdd.reserve(fraction);
    for (size_t keyIndex=from;keyIndex<to;keyIndex++){
        featuresToAdd.push_back(features[keysCopy[keyIndex]]);
        labelsToAdd.push_back(labels[keysCopy[keyIndex]]);
    }
    featuresFolds.push_back(featuresToAdd);
    labelsFolds.push_back(labelsToAdd);

    std::vector<std::vector<double>> trainFeatures;
    std::vector<int> trainLabels;
    std::vector<size_t> trainKeys;
    trainFeatures.reserve(length);
    trainLabels.reserve(length);
    trainKeys.reserve(length);
    for (size_t predictIndex=0;predictIndex<fold;predictIndex++){
        trainFeatures.clear();
        trainLabels.clear();
        trainKeys.clear();
        const std::vector<std::vector<double>>& predictFeatures = featuresFolds[predictIndex];
        const std::vector<int>& actualLabels = labelsFolds[predictIndex];
        std::vector<int> predictedLabels (actualLabels.size(),0);
        std::vector<size_t> predictKeys;
        for (size_t cnt = 0; cnt<actualLabels.size(); cnt++){
            predictKeys.push_back(cnt);
        }

        int counter = -1;
        for (size_t trainIndex = 0;trainIndex<fold;trainIndex++){
            if (trainIndex==predictIndex){
                continue;
            }
            size_t trainLength = featuresFolds[trainIndex].size();
            for (size_t it = 0; it<trainLength;it++){
                trainFeatures.push_back(featuresFolds[trainIndex][it]);
                trainLabels.push_back(labelsFolds[trainIndex][it]);
                trainKeys.push_back(++counter);
            }
        }

        Model submodel(trainFeatures,trainLabels,trainKeys,pixelCount);


/////////////////

        std::cout<<"Fold:"<<predictIndex<<std::endl;

        std::vector<std::thread> threads;
        threads.reserve(threadCount);

        size_t keysCount = predictKeys.size();
        size_t subfraction = keysCount / threadCount;

        auto itEnd = predictKeys.begin();
        for (size_t cnter = 0; cnter<threadCount-1;cnter++){
            auto itStart = predictKeys.begin() + subfraction*cnter;
            itEnd = itStart + subfraction;
            std::vector<size_t> part (itStart,itEnd);
            if(K<=0){
                threads.push_back(std::thread(predictorWeighted,&submodel,std::move(part),std::ref(predictFeatures),std::ref(predictedLabels),Ldist));
            }
            else {
                threads.push_back(std::thread(predictor,&submodel,std::move(part),std::ref(predictFeatures),std::ref(predictedLabels),Ldist,K));
            }
        }

         std::vector<size_t> part (itEnd,predictKeys.end());
        if(K<=0){
            threads.push_back(std::thread(predictorWeighted,&submodel,std::move(part),std::ref(predictFeatures),std::ref(predictedLabels),Ldist));
        }
        else {
            threads.push_back(std::thread(predictor,&submodel,std::move(part),std::ref(predictFeatures),std::ref(predictedLabels),Ldist,K));
        }

        for (auto& p : threads){
            p.join();
        }
    ////////////////////

        double correct = 0;
        size_t total = predictKeys.size();
        for (size_t index=0;index<total;index++){
            if (predictedLabels[index] == actualLabels[index]){
                correct++;
            }
        }
        double successRate = correct/total;
        accuracy.push_back(successRate);
    }
    double sum = 0;
    for(double sub : accuracy){
        sum += sub;
    }

    return (sum/fold);
}


static void writeDownResults(std::string filename,const std::vector<int>& data){

    std::fstream fileStream(filename.c_str(),std::ios::out|std::ios::trunc);
    std::string header = "ImageId,Label\n";
    fileStream.write(header.c_str(),strlen(header.c_str())*sizeof(char));


    size_t length = data.size();
    for (size_t index=0;index<length;index++){
        std::string line = std::to_string(index+1) + "," + std::to_string(data[index]) + "\n";
        fileStream.write(line.c_str(),strlen(line.c_str())*sizeof(char));
    }

    fileStream.close();

}






int main(const int argc, const char** argv){

//parse command line arguments start
    if (argc < 9){
        std::cout<<"Usage: knn TrainFilename UnlabeledFeaturesFilename Outputfile K Ldist threads Nfold components"<<std::endl;
        std::cout<<"Note: K, Ldist, threads, Nfold and Components are integers."<<std::endl;
        std::cout<<"Note: If Nfold is 0 then UnlabeledFeatures are predicted"<<std::endl;
        std::cout<<"Note: If K is 0 then weighted knn is used"<<std::endl;
        std::cout<<"Note: If components is 0 then all components are used"<<std::endl;
        exit (1);
    }

    std::string trainFilename (*(argv+1));
    std::string unlabeledFilename (*(argv+2));
    std::string outputFilename (*(argv+3));
    int argv4 = atoi (*(argv+4));
    int argv5 = atoi (*(argv+5));
    int argv6 = atoi(*(argv+6));
    int argv7 = atoi(*(argv+7));
    int argv8 = atoi(*(argv+8));

    if (argv4<0 || argv5<=0 || argv6<=0 || argv6>32 || argv7<0 || argv8<0){
        std::cout<<"K>=0 (0 for weighted KNN) Ldist>0 , threadsCount>0 ,threadsCount <=32 , Nfold>=0 ( (<2) for prediction of UnlableledFeaturesFilename) , componets>=0 ( 0 for all)"<<std::endl;
        exit (1);
    }

    size_t K = argv4;
    size_t Ldist = argv5;
    size_t threadsCount = argv6;
    size_t Nfold = argv7;
    size_t components = argv8;
    //parse command line arguments end


    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    Model M1 = Model(components);
    M1.train(trainFilename);

    if (Nfold<2){
        std::vector<int> results = M1.predict(unlabeledFilename,Ldist,threadsCount,K);
        writeDownResults(outputFilename,results);
    }
    else{
        double accuracy = M1.crossValidate(Nfold,Ldist,threadsCount,K);

        std::fstream fileStream(outputFilename,std::ios::out|std::ios::app);

        std::string line = "Fold=" + std::to_string(Nfold) + ", K=" + std::to_string(K) + ", L=" + std::to_string(Ldist) + ", accuracy="+std::to_string(accuracy)+"\n";
        fileStream.write(line.c_str(),strlen(line.c_str())*sizeof(char));

        fileStream.close();

    }

    std::chrono::high_resolution_clock::time_point finish = std::chrono::high_resolution_clock::now();
    std::chrono::seconds time_span = std::chrono::duration_cast<std::chrono::seconds>(finish - start);
    std::cout<<time_span.count()<<std::endl;

    return 0;
}
